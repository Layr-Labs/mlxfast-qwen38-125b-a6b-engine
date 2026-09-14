// TrackQuantPrologue.swift -- fused RMSNorm + activation packing + inject GEMV.
//
// Decode only, one shape: the post-mixer inject-path GEMV
// (N = 4, K = hc_count * hidden = 10240, affine-4 g32).
//
// Today that chain is three launches MLX compile cannot merge (RMSNorm is a
// CustomKernel; CustomKernel is off the fuse whitelist): grouped RMSNorm,
// the bf16 store of the scaled activations, then `qmv`'s small-N branch.
// This file is one Metal kernel whose prologue computes the RMSNorm in f32
// from the raw residual and the norm weights, packs each activation tile in
// registers with `load_vector`'s 4-bit layout, and feeds those tiles straight
// into the MAC loop.
//
// Numerics are the existing formulas in the existing order:
//   1. residual add in the array dtype (bf16 mul + add) when HAS_INJECT
//   2. mean of squares in f32, `precise::rsqrt(mean + eps)`, T-round of
//      `x * inv_mean`, then T-multiply by the norm scale
//   3. `load_vector` packing (`/16 /256 /4096`) and `qdot` (scale * accum +
//      sum * bias) with the same lane-to-K map as `track_inject_qmv`
// No K-walk reassociation. Bit-exact vs the three-kernel reference, so
// `TRACK_QUANT_PROLOGUE` defaults ON. Toggle-off or a shape mismatch falls
// back to the separate kernels.
//
// The down GEMV (320 rows) still reads the stored `normed` vector; only the
// 4-row inject GEMV consumes the register tiles. A second shape did not fall
// out of the same helper without rewriting `qmv_fast_reg`.

import Foundation
import MLX
import MLXFast

enum TrackQuantPrologue {
    /// Kill switch. Unset or any value other than "0" keeps the fused kernel.
    static let enabled =
        ProcessInfo.processInfo.environment["TRACK_QUANT_PROLOGUE"] != "0"

    /// Production inject-path geometry (Qwen 3.8 125B A6B decode).
    static let hidden = 2560
    static let hcCount = 4
    static let injectRows = 4
    static let groupSize = 32
    static let bits = 4

    static func shouldApply(
        s: Int, k: Int, n: Int, hidden: Int, hc: Int,
        bits: Int, groupSize: Int, hasInjectWeight: Bool,
        enabled: Bool = enabled
    ) -> Bool {
        enabled && hasInjectWeight && s == 1 && n == injectRows && hc == hcCount
            && hidden == Self.hidden && k == hidden * hc && bits == Self.bits
            && groupSize == Self.groupSize
    }

    // MARK: - CPU numeric core (no Metal)

    /// Round to bfloat16, IEEE nearest-even. Matches Metal `static_cast<bfloat16_t>`.
    static func roundBF16(_ x: Float) -> Float {
        if x.isNaN || x.isInfinite { return x }
        let u = x.bitPattern
        let lsb = (u >> 16) & 1
        return Float(bitPattern: (u &+ (0x7FFF &+ lsb)) & 0xFFFF_0000)
    }

    struct PackedWeight {
        var w: [UInt32]
        var scales: [Float]
        var biases: [Float]
        var rows: Int
        var k: Int
    }

    /// Affine-4, group 32. Same scale/bias formula as `affine_quantize`
    /// (`quantized.h`): max init 0, `max((max-min)/15, 1e-7)`, side from
    /// `|min| > |max|`, `q0 = round(edge/scale)`, rescale unless q0 == 0.
    static func affineQuantize4(_ dense: [Float], rows: Int, k: Int) -> PackedWeight {
        precondition(k % 32 == 0 && dense.count == rows * k)
        let groups = k / 32
        var w = [UInt32](repeating: 0, count: rows * (k / 8))
        var scales = [Float](repeating: 0, count: rows * groups)
        var biases = [Float](repeating: 0, count: rows * groups)
        let nBins: Float = 15
        let qeps: Float = 1e-7
        for row in 0..<rows {
            for g in 0..<groups {
                let base = row * k + g * 32
                var wMin = Float.greatestFiniteMagnitude
                var wMax: Float = 0
                for i in 0..<32 {
                    let v = dense[base + i]
                    wMin = min(wMin, v)
                    wMax = max(wMax, v)
                }
                var scale = max((wMax - wMin) / nBins, qeps)
                let side = abs(wMin) > abs(wMax)
                if !side { scale = -scale }
                let edge = side ? wMin : wMax
                let q0 = (edge / scale).rounded(.toNearestOrAwayFromZero)
                let atZero = q0 == 0
                if !atZero { scale = edge / q0 }
                let bias: Float = atZero ? 0 : edge
                scales[row * groups + g] = scale
                biases[row * groups + g] = bias
                for i in 0..<32 {
                    let q = ((dense[base + i] - bias) / scale)
                        .rounded(.toNearestOrAwayFromZero)
                    let qi = UInt32(min(max(q, 0), nBins))
                    let kIndex = g * 32 + i
                    w[row * (k / 8) + kIndex / 8] |= (qi & 0xF) << (4 * (kIndex % 8))
                }
            }
        }
        return PackedWeight(w: w, scales: scales, biases: biases, rows: rows, k: k)
    }

    /// Grouped RMSNorm matching `track_inject_norm`: per-hc mean of squares in
    /// f32, rsqrt(mean + eps), T-round of `x * inv`, then T-multiply by scale.
    static func rmsNormCPU(
        residual: [Float], out: [Float]?, injectGate: [Float]?,
        normW: [Float], hidden: Int, hc: Int, eps: Float, tile: Bool
    ) -> (stream: [Float], normed: [Float], invMean: [Float]) {
        let k = hidden * hc
        precondition(normW.count == k)
        var stream = [Float](repeating: 0, count: k)
        var normed = [Float](repeating: 0, count: k)
        var invMean = [Float](repeating: 0, count: hc)
        for h in 0..<hc {
            var acc: Float = 0
            for d in 0..<hidden {
                var r = tile ? residual[d] : residual[h * hidden + d]
                if let out, let gate = injectGate {
                    let sp = roundBF16(out[d] * gate[h])
                    r = roundBF16(r + sp)
                }
                stream[h * hidden + d] = r
                acc += r * r
            }
            let inv = 1 / (acc / Float(hidden) + eps).squareRoot()
            invMean[h] = inv
            for d in 0..<hidden {
                let n = roundBF16(stream[h * hidden + d] * inv)
                normed[h * hidden + d] = roundBF16(n * normW[h * hidden + d])
            }
        }
        return (stream, normed, invMean)
    }

    /// `qmv_impl` small-N (N < 8) for affine-4: 32 lanes, block 256, packs_per_thread
    /// = 1, `load_vector` + `qdot`, sequential fold of the 32 lane partials in
    /// place of `simd_sum`. Both the fused path and the three-kernel reference
    /// use this, so the CPU comparison is bit-exact under one association.
    static func qmvSmallNCPU(x: [Float], packed: PackedWeight) -> [Float] {
        let n = packed.rows, k = packed.k
        precondition(k % 8 == 0 && x.count == k)
        let block = 256
        let vpt = 8
        let nFull = (k - 1) / block
        var y = [Float](repeating: 0, count: n)
        for row in 0..<n {
            var laneAcc = [Float](repeating: 0, count: 32)
            for lid in 0..<32 {
                var acc: Float = 0
                for i in 0..<nFull {
                    let k0 = i * block + lid * vpt
                    acc += qdotBlock(x: x, k0: k0, packed: packed, row: row, i: i, lid: lid)
                }
                let kEnd = nFull * block
                let remaining = min(max(k - kEnd - lid * vpt, 0), vpt)
                if remaining > 0 {
                    acc += qdotBlock(x: x, k0: kEnd + lid * vpt, packed: packed, row: row, i: nFull, lid: lid)
                }
                laneAcc[lid] = acc
            }
            var total: Float = 0
            for a in laneAcc { total += a }
            y[row] = roundBF16(total)
        }
        return y
    }

    /// Three-kernel reference: RMSNorm (store stream + normed) then qmv on the
    /// stored activations.
    static func threeKernelCPU(
        residual: [Float], out: [Float]?, injectGate: [Float]?,
        normW: [Float], packed: PackedWeight,
        hidden: Int, hc: Int, eps: Float, tile: Bool
    ) -> (stream: [Float], normed: [Float], inj: [Float]) {
        let r = rmsNormCPU(
            residual: residual, out: out, injectGate: injectGate,
            normW: normW, hidden: hidden, hc: hc, eps: eps, tile: tile)
        let inj = qmvSmallNCPU(x: r.normed, packed: packed)
        return (r.stream, r.normed, inj)
    }

    /// Fused path: same RMSNorm, then qmv tiles rebuilt from stream + inv_mean
    /// + normW in "registers" (no load of the stored normed vector).
    static func fusedCPU(
        residual: [Float], out: [Float]?, injectGate: [Float]?,
        normW: [Float], packed: PackedWeight,
        hidden: Int, hc: Int, eps: Float, tile: Bool
    ) -> (stream: [Float], normed: [Float], inj: [Float]) {
        let r = rmsNormCPU(
            residual: residual, out: out, injectGate: injectGate,
            normW: normW, hidden: hidden, hc: hc, eps: eps, tile: tile)
        var tiles = [Float](repeating: 0, count: r.normed.count)
        for d in 0..<r.stream.count {
            let hcIdx = d / hidden
            let n = roundBF16(r.stream[d] * r.invMean[hcIdx])
            tiles[d] = roundBF16(n * normW[d])
        }
        let inj = qmvSmallNCPU(x: tiles, packed: packed)
        return (r.stream, r.normed, inj)
    }

    // MARK: - Metal

    static let header =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
        + TrackFastKernels.mixerHeadHeaderTail + #"""

        // load_vector for bits=4, values_per_thread=8, but the 8 activations are
        // RMSNormed from `stream` in registers: T(float(r) * inv) * normW, then
        // the same /16 /256 /4096 packing qdot expects. Never a global load of
        // the stored `normed` vector.
        template <typename T, int H>
        METAL_FUNC float load_vector_norm4(
            const device T* stream,
            const device T* normW,
            threadgroup float* inv_mean,
            int k0,
            thread float* x_thread) {
          typedef float U;
          U sum = 0;
          #pragma unroll
          for (int i = 0; i < 8; i += 4) {
            U a[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int d = k0 + i + j;
              const T n = static_cast<T>(float(stream[d]) * inv_mean[d / H]);
              a[j] = static_cast<U>(n * normW[d]);
            }
            sum += a[0] + a[1] + a[2] + a[3];
            x_thread[i] = a[0];
            x_thread[i + 1] = a[1] / 16.0f;
            x_thread[i + 2] = a[2] / 256.0f;
            x_thread[i + 3] = a[3] / 4096.0f;
          }
          return sum;
        }

        // track_inject_qmv with load_vector replaced by load_vector_norm4.
        // Same lanes, same K walk, same qdot, same simd_sum.
        template <typename T, int group_size, int bits, int in_vec_size, int out_vec_size, int H, int UNR>
        METAL_FUNC void track_inject_qmv_norm(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* stream,
            const device T* normW,
            threadgroup float* inv_mean,
            device T* y,
            uint simd_gid,
            uint simd_lid) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(bits == 4, "inject prologue is affine-4");
          static_assert(values_per_thread == 8, "load_vector_norm4");
          static_assert(out_vec_size < num_simdgroups * results_per_simdgroup, "small-N");
          static_assert(in_vec_size > block_size, "K walk");
          static_assert(in_vec_size % values_per_thread == 0, "exact tail");

          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          thread U result[results_per_simdgroup] = {0};

          constexpr int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          constexpr int in_vec_size_g = in_vec_size / group_size;
          const int out_row = simd_gid * results_per_simdgroup;
          if (out_row >= out_vec_size) {
            return;
          }
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          y += out_row;

          constexpr int NFULL = (in_vec_size - 1) / block_size;
          constexpr int NR = out_vec_size < results_per_simdgroup ? out_vec_size : results_per_simdgroup;
          #pragma clang loop unroll_count(UNR)
          for (int i = 0; i < NFULL; i++) {
            U sum = load_vector_norm4<T, H>(
                stream, normW, inv_mean, i * block_size + (int)simd_lid * values_per_thread, x_thread);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
          }
          constexpr int k_end = NFULL * block_size;
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k_end - simd_lid * values_per_thread), 0, values_per_thread);
          if (remaining > 0) {
            U sum = load_vector_norm4<T, H>(
                stream, normW, inv_mean, k_end + (int)simd_lid * values_per_thread, x_thread);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
          }
          for (int row = 0; row < NR; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) {
              y[row] = static_cast<T>(result[row]);
            }
          }
        }
        """#

    /// Prologue: injectNormWide layout (one simdgroup plays the 640-thread
    /// rms_single_row, same lane, same simd_sum) for two HC groups per
    /// simdgroup. Then the register-fed inject qmv.
    static let source = """
        constexpr int N_READS = 4;
        constexpr uint NT = H / N_READS;
        constexpr uint rms_sgs = (H + 32 * N_READS - 1) / (32 * N_READS);
        const uint row = thread_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint simd_gid = sg;
        const uint simd_lid = lane;
        threadgroup float local_sums[2][32];
        threadgroup float inv_mean_hc[HC];
        const uint W = HC * H;

        for (int hci = 0; hci < 2; ++hci) {
            const uint hc = sg * 2 + (uint)hci;
            const uint base = row * W + hc * H;
            T inj_t = T(0);
            if (HAS_INJECT) { inj_t = injectGate[row * HC + hc]; }
            for (uint g = 0; g < rms_sgs; ++g) {
                const uint lid = g * 32 + lane;
                float acc = 0.0f;
                if (lid < NT) {
                    for (int i = 0; i < N_READS; ++i) {
                        const uint d = lid * N_READS + i;
                        const uint src = TILE ? (row * H + d) : (base + d);
                        T r = residual[src];
                        if (HAS_INJECT) {
                            T sp = out[row * H + d] * inj_t;
                            r = r + sp;
                        }
                        stream[base + d] = r;
                        const float xf = static_cast<float>(r);
                        acc += xf * xf;
                    }
                }
                acc = simd_sum(acc);
                if (lane == 0) { local_sums[sg][g] = acc; }
            }
            if (lane >= rms_sgs) { local_sums[sg][lane] = 0; }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const float total = simd_sum(local_sums[sg][lane]);
            const float inv = metal::precise::rsqrt(total / (float)H + as_type<float>((uint)EPS_BITS));
            if (lane == 0) { inv_mean_hc[hc] = inv; }
            for (uint g = 0; g < rms_sgs; ++g) {
                const uint lid = g * 32 + lane;
                if (lid < NT) {
                    for (int i = 0; i < N_READS; ++i) {
                        const uint d = lid * N_READS + i;
                        T n = static_cast<T>(static_cast<float>(stream[base + d]) * inv);
                        normed[base + d] = n * normW[hc * H + d];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const device T* sr = stream + (size_t)row * (size_t)W;
        const device T* nr = normW;
        device T* yr = inj + (size_t)row * (size_t)HC;
        track_inject_qmv_norm<T, GS, BITS, KD, HC, H, UNR>(
            iw, isc, ib, sr, nr, inv_mean_hc, yr, simd_gid, simd_lid);
        """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_inject_qmv_norm_prologue",
        inputNames: ["residual", "out", "injectGate", "normW", "iw", "isc", "ib"],
        outputNames: ["stream", "normed", "inj"],
        source: source, header: header, ensureRowContiguous: true)

    /// Fused inject GEMV, or nil when the toggle is off, the shape is not the
    /// decode inject path, or the default stream is not the GPU.
    static func apply(
        residual: MLXArray, out: MLXArray?, injectGate: MLXArray?,
        scale: MLXArray, weight: TrackQuantWeight,
        hcCount: Int, hidden: Int, eps: Float, tile: Bool
    ) -> (stream: MLXArray, normed: MLXArray, inj: MLXArray)? {
        let B = residual.dim(0), S = residual.dim(1)
        let k = hcCount * hidden
        guard shouldApply(
            s: S, k: k, n: weight.rows, hidden: hidden, hc: hcCount,
            bits: weight.bits, groupSize: weight.groupSize,
            hasInjectWeight: weight.biases != nil),
            StreamOrDevice.default.stream === Stream.gpu,
            residual.dtype == .bfloat16,
            weight.mode == .affine, weight.weight.dtype == .uint32,
            let biases = weight.biases
        else { return nil }
        let hasInject = out != nil
        let outs = kernel(
            [
                residual, out ?? residual, injectGate ?? residual, scale,
                weight.weight, weight.scales, biases,
            ],
            template: [
                ("T", residual.dtype), ("GS", weight.groupSize), ("BITS", weight.bits),
                ("H", hidden), ("HC", hcCount), ("KD", k),
                ("HAS_INJECT", hasInject), ("TILE", tile),
                ("EPS_BITS", Int(eps.bitPattern)), ("UNR", 4),
            ],
            grid: (64, B * S, 1), threadGroup: (64, 1, 1),
            outputShapes: [[B, S, k], [B, S, k], [B, S, hcCount]],
            outputDTypes: [residual.dtype, residual.dtype, residual.dtype])
        return (outs[0], outs[1], outs[2])
    }

    // MARK: - qdot block

    private static func qdotBlock(
        x: [Float], k0: Int, packed: PackedWeight, row: Int, i: Int, lid: Int
    ) -> Float {
        var xv = [Float](repeating: 0, count: 8)
        for t in 0..<8 { xv[t] = x[k0 + t] }
        var xt = [Float](repeating: 0, count: 8)
        var sum: Float = 0
        for t in stride(from: 0, to: 8, by: 4) {
            sum += xv[t] + xv[t + 1] + xv[t + 2] + xv[t + 3]
            xt[t] = xv[t]
            xt[t + 1] = xv[t + 1] / 16
            xt[t + 2] = xv[t + 2] / 256
            xt[t + 3] = xv[t + 3] / 4096
        }
        let word = packed.w[row * (packed.k / 8) + i * 32 + lid]
        let gi = row * (packed.k / 32) + i * 8 + lid / 4
        let w0 = UInt16(truncatingIfNeeded: word)
        let w1 = UInt16(truncatingIfNeeded: word >> 16)
        var accum: Float = 0
        accum += xt[0] * Float(w0 & 0x000F)
        accum += xt[1] * Float(w0 & 0x00F0)
        accum += xt[2] * Float(w0 & 0x0F00)
        accum += xt[3] * Float(w0 & 0xF000)
        accum += xt[4] * Float(w1 & 0x000F)
        accum += xt[5] * Float(w1 & 0x00F0)
        accum += xt[6] * Float(w1 & 0x0F00)
        accum += xt[7] * Float(w1 & 0xF000)
        return packed.scales[gi] * accum + sum * packed.biases[gi]
    }
}
