// Residual-add GEMV epilogue: y = GEMV + residual, with the hyper-connection
// inject form stream = residual + GEMV * inject. Default ON
// (`TRACK_RESIDUAL_EPILOGUE` unset or "1"). Unfit shapes keep the separate add
// inside `injectNorm`.

import Foundation
import MLX
import MLXFast

enum TrackResidualEpilogue {
    static let enabled =
        (ProcessInfo.processInfo.environment["TRACK_RESIDUAL_EPILOGUE"] ?? "1") != "0"

    static func fitsOutProj(k: Int, n: Int, tokens: Int) -> Bool {
        tokens >= 1 && tokens <= 8 && k == 6144 && n == 2560
    }

    static func fitsDownProj(k: Int, n: Int, tokens: Int) -> Bool {
        tokens >= 1 && tokens <= 8 && k == 640 && n == 2560
    }

    static func coversOutProj(k: Int, n: Int, tokens: Int) -> Bool {
        enabled && fitsOutProj(k: k, n: n, tokens: tokens)
    }

    static func coversDownProj(k: Int, n: Int, tokens: Int) -> Bool {
        enabled && fitsDownProj(k: k, n: n, tokens: tokens)
    }

    /// Round float32 to bf16 bits, RNTE. Inf and NaN keep the high 16 bits.
    static func store(_ x: Float) -> UInt16 {
        let u = x.bitPattern
        if u & 0x7FFF_FFFF >= 0x7F80_0000 {
            return UInt16(truncatingIfNeeded: u >> 16)
        }
        let lsb = (u >> 16) & 1
        return UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ lsb) >> 16)
    }

    static func load(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    /// GEMV already rounded to bf16, then f32 add with residual, then bf16 store.
    static func add(gemvF32: Float, residual: UInt16) -> UInt16 {
        return store(load(store(gemvF32)) + load(residual))
    }

    static func injectAdd(gemvF32: Float, residual: UInt16, inject: UInt16) -> UInt16 {
        let g = store(gemvF32)
        let sp = store(load(g) * load(inject))
        return store(load(residual) + load(sp))
    }

    static func injectAdd(
        gemvF32: [Float], residual: [UInt16], inject: [UInt16], hidden: Int
    ) -> [UInt16] {
        let hc = inject.count
        var out = [UInt16](repeating: 0, count: hc * hidden)
        for d in 0..<hidden {
            let g = store(gemvF32[d])
            for h in 0..<hc {
                let sp = store(load(g) * load(inject[h]))
                out[h * hidden + d] = store(load(residual[h * hidden + d]) + load(sp))
            }
        }
        return out
    }

    // MARK: quantized GEMV + inject residual (o_proj / GDN out_proj)

    /// Decode o_proj / GDN out_proj: K=6144 qmv_fast, N=2560, HC streams.
    /// stream[s, hc, d] = residual[s, hc, d] + T(gemv[s, d]) * inject[s, hc]
    static let source = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        if constexpr (VPT == 1) {
            float r[4];
            const int row0 = tile * (SG * 4) + (int)sg * 4;
            qmv_fast_reg<T, GS, BITS, 4>(w, scales, biases, x, K, row0, lid, r);
            if (lid == 0) {
                for (int i = 0; i < 4; ++i) {
                    const int d = row0 + i;
                    if (d >= H) { continue; }
                    const T g = static_cast<T>(r[i]);
                    for (int hc = 0; hc < HC; ++hc) {
                        const T sp = g * inj[hc];
                        const T rv = residual[(size_t)hc * (size_t)H + (size_t)d];
                        stream[(size_t)hc * (size_t)H + (size_t)d] = rv + sp;
                    }
                }
            }
        } else {
            const int row = tile * 8 + (int)sg * 4 + (int)(lid / 8);
            float rw[VPT];
            qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(w, scales, biases, x, K, VPT, row, lid, rw);
            if ((lid % 8) == 0) {
                for (int v = 0; v < VPT; ++v) {
                    const T g = static_cast<T>(rw[v]);
                    for (int hc = 0; hc < HC; ++hc) {
                        const T sp = g * inj[(size_t)v * (size_t)HC + (size_t)hc];
                        const T rv = residual[(size_t)v * (size_t)(HC * H)
                            + (size_t)hc * (size_t)H + (size_t)row];
                        stream[(size_t)v * (size_t)(HC * H)
                            + (size_t)hc * (size_t)H + (size_t)row] = rv + sp;
                    }
                }
            }
        }
        """

    static let kernel = MLXFast.metalKernel(
        name: "track_qmv_residual",
        inputNames: ["x", "w", "scales", "biases", "residual", "inj"],
        outputNames: ["stream"],
        source: source, header: TrackFastMixerKernels.header, ensureRowContiguous: true)
    static let kernel1 = MLXFast.metalKernel(
        name: "track_qmv_residual_1",
        inputNames: ["x", "w", "scales", "biases", "residual", "inj"],
        outputNames: ["stream"],
        source: source, header: TrackFastMixerKernels.header1, ensureRowContiguous: true)

    /// S=1 o_proj / GDN out_proj: simdgroups per threadgroup. 2 is the tip.
    /// SG=1 passed local teacher-forced 64 and mismatched ranked free-run at
    /// step 2 (Yukon f61e5a / 5fadbfe).
    static let decodeSimdgroupsPerThreadgroup = 2

    /// MLX `grid` is threads. Y threads = threadgroups * tgY.
    static func decodeLaunch(hidden: Int, tokens: Int) -> (
        gridY: Int, tgY: Int, rowsPerThreadgroup: Int
    ) {
        let sg = tokens == 1 ? decodeSimdgroupsPerThreadgroup : 2
        let rowsPerTg = sg * 4
        return ((hidden / rowsPerTg) * sg, sg, rowsPerTg)
    }

    /// First output row owned by `(tile, sg)` under `sgPerTg` simdgroups/TG.
    static func decodeRow0(tile: Int, sg: Int, sgPerTg: Int) -> Int {
        tile * (sgPerTg * 4) + sg * 4
    }

    /// Returns the W-wide stream when the projection, window and residual
    /// match the decode o_proj / GDN out_proj gate. Nil keeps `injectNorm`.
    static func project(
        _ proj: TrackProj, x: MLXArray, residual: MLXArray, inject: MLXArray,
        hcCount: Int, hidden: Int
    ) -> MLXArray? {
        guard enabled, StreamOrDevice.default.stream === Stream.gpu,
            x.dtype == .bfloat16, residual.dtype == x.dtype, inject.dtype == x.dtype,
            case .quant(let q) = proj, q.bits == 4, q.groupSize == 32, q.mode == .affine,
            q.weight.dtype == .uint32, q.scales.dtype == .bfloat16,
            let biases = q.biases, biases.dtype == .bfloat16
        else { return nil }
        let n = q.rows
        let k = q.weight.dim(1) * 8
        let S = x.dim(x.ndim - 2)
        let B = x.ndim >= 3 ? x.dim(0) : 1
        guard B == 1, x.dim(-1) == k, n == hidden,
            fitsOutProj(k: k, n: n, tokens: S),
            residual.dim(-1) == hcCount * hidden, inject.dim(-1) == hcCount,
            residual.dim(residual.ndim - 2) == S, inject.dim(inject.ndim - 2) == S,
            n % 8 == 0, k % 512 == 0
        else { return nil }
        let x2 = x.reshaped(S, k)
        let res2 = residual.reshaped(S, hcCount * hidden)
        let inj2 = inject.reshaped(S, hcCount)
        let launch = decodeLaunch(hidden: n, tokens: S)
        let outs = (S == 1 ? kernel1 : kernel)(
            [x2, q.weight, q.scales, biases, res2, inj2],
            template: [
                ("T", x.dtype), ("GS", q.groupSize), ("BITS", q.bits),
                ("K", k), ("H", hidden), ("HC", hcCount), ("VPT", S),
                ("SG", launch.tgY),
            ],
            grid: (32, launch.gridY, 1), threadGroup: (32, launch.tgY, 1),
            outputShapes: [[S, hcCount * hidden]],
            outputDTypes: [x.dtype])
        return outs[0].reshaped(1, S, hcCount * hidden)
    }
}
