// TrackPLEChain.swift -- bit-exact PLE per-step dispatch fusion.
//
// One-token decode used to issue the original PLE op chain (~22 launches
// after the n-gram rows, plus a piecewise dilated conv that fell through to
// implicit GEMM with groups = W) and, on the device hash path, a long
// shift/xor/mod graph for the 16-head n-gram lookup.
//
// TRACK_PLE_CHAIN_FUSE (default ON) merges the mergeable stages without
// changing values:
//   key_proj, value_proj, prepare (both RMS, InT product, row_reduce_looped
//   sum, gate chain, gated value, conv RMS, concat), conv (dilated taps +
//   silu + residual). That is 4 launches. The n-gram hash's piecewise
//   shift/xor/mod graph becomes one integer kernel on the device path (host
//   decode already hashes on the CPU and issues 0).
//
// Kill switch TRACK_PLE_CHAIN_FUSE=0 restores TrackFastPLEKernels (prod +
// MLX sum + gated + concat + conv). Shape/guard misses fall back the same
// way. TrackPLEFusion (PLEFUSE2) changes reduction association; it is
// §5.4-priced and is not selected.

import Foundation
import MLX
import MLXLLM

enum TrackPLEChain {

    static var isEnabled: Bool {
        (ProcessInfo.processInfo.environment["TRACK_PLE_CHAIN_FUSE"] ?? "1") != "0"
    }

    /// Cap the fused PLE *compute* graph is held to, after n-gram rows exist.
    static let fusedComputeDispatchCap = 4

    /// Original PLE block after n-gram rows: one launch per named stage.
    static let unfusedCompute: [String] = [
        "key_proj", "norm_key_rms", "norm_key_scale",
        "value_proj", "norm_query_rms", "norm_query_scale",
        "mul_kq", "sum_dot", "div_sqrt_h",
        "abs", "maximum", "sqrt", "sign", "sign_mul",
        "sigmoid", "mul_value",
        "norm_conv_rms", "norm_conv_scale",
        "concat_state", "conv1d_implicit_gemm", "silu", "residual_add",
    ]

    /// Merged compute graph. Two projections stay the vendor qmm; prepare
    /// and conv are one launch each.
    static let fusedCompute: [String] = [
        "key_proj", "value_proj", "prepare", "conv",
    ]

    /// Device-side n-gram `rowIds` before fusion: three shift-right copies
    /// (shift 0 is a view) plus mix/mod/concat. Host decode issues 0.
    static let unfusedNgramHash: [String] = [
        "concat_history", "cast_i64",
        "shift1_eos_where", "shift1_cummax", "shift1_concat", "shift1_in_segment",
        "shift1_source", "shift1_clamp", "shift1_take", "shift1_usable", "shift1_where",
        "shift2_eos_where", "shift2_cummax", "shift2_concat", "shift2_in_segment",
        "shift2_source", "shift2_clamp", "shift2_take", "shift2_usable", "shift2_where",
        "ngram2_mul", "ngram2_mod", "ngram2_add",
        "ngram3_mul", "ngram3_xor", "ngram3_mul", "ngram3_mod", "ngram3_add",
        "concat_heads",
    ]

    static let fusedNgramHash: [String] = ["ngram_ids"]

    static func computeSchedule(fused: Bool) -> [String] {
        fused ? fusedCompute : unfusedCompute
    }

    static func ngramHashSchedule(fused: Bool, hostPath: Bool) -> [String] {
        if hostPath { return [] }
        return fused ? fusedNgramHash : unfusedNgramHash
    }

    static func shouldFuse(
        enabled: Bool = isEnabled,
        S: Int,
        hidden: Int,
        hcCount: Int,
        dilation: Int,
        stateLength: Int,
        dtype: DType,
        keyRows: Int,
        valueRows: Int,
        convCols: Int
    ) -> Bool {
        enabled
            && S >= 1 && S <= 8
            && hidden >= 128 && hidden % 4 == 0 && hidden / 4 <= 1024
            && hcCount >= 1
            && dilation >= 1
            && stateLength == (convCols - 1) * dilation
            && keyRows == hcCount * hidden
            && valueRows == hidden
            && convCols >= 2
            && [.bfloat16, .float16, .float32].contains(dtype)
    }

    // MARK: - Fused prepare (prod + row_reduce_looped sum + gated + concat)

    static let prepareSource = """
        constexpr int N_READS = 4;
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        const uint lid = thread_position_in_threadgroup.x;
        const uint hc = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float partials[32];
        threadgroup InT dpartials[32];
        const uint S_ = S;
        const uint b = row / S_;
        const uint s = row % S_;
        const uint base = row * W + hc * H;

        float kx[N_READS];
        float qx[N_READS];
        float kacc = 0.0f;
        float qacc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            kx[i] = static_cast<float>(keyFlat[base + d]);
            qx[i] = static_cast<float>(stream[base + d]);
            kacc += kx[i] * kx[i];
            qacc += qx[i] * qx[i];
        }
        kacc = simd_sum(kacc);
        qacc = simd_sum(qacc);
        if (sg == 0 && lane >= simd_groups) { partials[lane] = 0; }
        if (lane == 0) { partials[sg] = kacc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kacc = simd_sum(partials[lane]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane >= simd_groups) { partials[lane] = 0; }
        if (lane == 0) { partials[sg] = qacc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        qacc = simd_sum(partials[lane]);
        const float kinv = metal::precise::rsqrt(kacc / (float)H + eps);
        const float qinv = metal::precise::rsqrt(qacc / (float)H + eps);

        InT accp = InT(0);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT kn = static_cast<InT>(kx[i] * kinv) * kscale[hc * H + d];
            const InT qn = static_cast<InT>(qx[i] * qinv) * qscale[hc * H + d];
            accp = (kn * qn) + accp;
        }
        accp = simd_sum(accp);
        if (sg == 0 && lane >= simd_groups) { dpartials[lane] = InT(0); }
        if (lane == 0) { dpartials[sg] = accp; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        accp = simd_sum(dpartials[lane]);

        InT g = accp / divisor;
        g = mlx_sqrt_t(mlx_maximum(mlx_abs_t(g), floorv)) * mlx_sign(g);
        const InT sgm = mlx_sigmoid(g);

        float gx[N_READS];
        float vacc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT v = sgm * value[(b * S_ + s) * H + d];
            gated[base + d] = v;
            gx[i] = static_cast<float>(v);
            vacc += gx[i] * gx[i];
        }
        vacc = simd_sum(vacc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane >= simd_groups) { partials[lane] = 0; }
        if (lane == 0) { partials[sg] = vacc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        vacc = simd_sum(partials[lane]);
        const float inv = metal::precise::rsqrt(vacc / (float)H + eps);
        const uint nfull = STATE_N + S_;
        const uint fullBase = (b * nfull + STATE_N + s) * W + hc * H;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT n = static_cast<InT>(gx[i] * inv) * cscale[hc * H + d];
            full[fullBase + d] = n;
        }
        if (s == 0) {
            for (uint t = 0; t < STATE_N; ++t) {
                const uint dst = (b * nfull + t) * W + hc * H;
                const uint src = (b * STATE_N + t) * W + hc * H;
                for (int i = 0; i < N_READS; ++i) {
                    const uint d = lid * N_READS + i;
                    full[dst + d] = convState[src + d];
                }
            }
        }
        """

    nonisolated(unsafe) static let prepareKernel = MLXFast.metalKernel(
        name: "track_ple_chain_prepare",
        inputNames: [
            "keyFlat", "stream", "value", "kscale", "qscale", "cscale",
            "convState", "divisor", "floorv", "eps",
        ],
        outputNames: ["gated", "full"],
        source: prepareSource,
        header: TrackFastPLEKernels.header + """
            template <typename T> METAL_FUNC T mlx_abs_t(T x) { return metal::abs(x); }
            template <typename T> METAL_FUNC T mlx_sqrt_t(T x) { return metal::precise::sqrt(x); }
            """,
        ensureRowContiguous: true)

    static func prepare(
        keyFlat: MLXArray, stream: MLXArray, value: MLXArray,
        kScale: MLXArray, qScale: MLXArray, cScale: MLXArray,
        convState: MLXArray, hcCount: Int, hidden: Int, eps: Float
    ) -> (gated: MLXArray, full: MLXArray) {
        let B = keyFlat.dim(0), S = keyFlat.dim(1), W = hcCount * hidden
        let stateN = convState.dim(1)
        let dtype = keyFlat.dtype
        let divisor = Foundation.sqrt(Float(hidden)).asMLXArray(dtype: dtype)
        let floor = TrackFastKernels.scalar(Float(1e-6), dtype: dtype)
        let outs = prepareKernel(
            [
                keyFlat, stream, value, kScale, qScale, cScale, convState,
                divisor, floor, MLXArray(eps),
            ],
            template: [
                ("InT", dtype), ("H", hidden), ("W", W), ("HC", hcCount),
                ("S", S), ("STATE_N", stateN),
            ],
            grid: (hidden / 4, hcCount, B * S), threadGroup: (hidden / 4, 1, 1),
            outputShapes: [[B, S, W], [B, stateN + S, W]],
            outputDTypes: [dtype, dtype])
        return (outs[0], outs[1])
    }

    /// 4-launch compute: the two projections are applied by the caller.
    static func afterProjections(
        keyFlat: MLXArray, stream: MLXArray, value: MLXArray,
        kScale: MLXArray, qScale: MLXArray, cScale: MLXArray,
        convState: MLXArray, convW: MLXArray, dilation: Int,
        hcCount: Int, hidden: Int, eps: Float
    ) -> (full: MLXArray, output: MLXArray) {
        let gn = prepare(
            keyFlat: keyFlat, stream: stream, value: value,
            kScale: kScale, qScale: qScale, cScale: cScale,
            convState: convState, hcCount: hcCount, hidden: hidden, eps: eps)
        let output = TrackFastPLEKernels.conv(
            full: gn.full, convW: convW, gated: gn.gated, dilation: dilation)
        return (gn.full, output)
    }

    // MARK: - Unfused compute (kill-switch / exactness oracle)

    static func unfusedAfterProjections(
        keyFlat: MLXArray, stream: MLXArray, value: MLXArray,
        kScale: MLXArray, qScale: MLXArray, cScale: MLXArray,
        convState: MLXArray, convW: MLXArray, dilation: Int,
        hcCount: Int, hidden: Int, eps: Float
    ) -> (full: MLXArray, output: MLXArray) {
        let B = keyFlat.dim(0), S = keyFlat.dim(1)
        let prod = TrackFastPLEKernels.prod(
            keyFlat: keyFlat, stream: stream, kScale: kScale, qScale: qScale,
            hcCount: hcCount, hidden: hidden, eps: eps)
        let dot = prod.reshaped(B, S, hcCount, hidden).sum(axis: -1, keepDims: true)
        let divisor = Foundation.sqrt(Float(hidden)).asMLXArray(dtype: dot.dtype)
        let floor = TrackFastKernels.scalar(Float(1e-6), dtype: dot.dtype)
        let gn = TrackFastPLEKernels.gated(
            g0: dot, value: value, cScale: cScale, divisor: divisor, floor: floor,
            hcCount: hcCount, hidden: hidden, eps: eps)
        let full = concatenated([convState, gn.normed], axis: 1)
        let output = TrackFastPLEKernels.conv(
            full: full, convW: convW, gated: gn.gated, dilation: dilation)
        return (full, output)
    }

    // MARK: - Fused n-gram hash (device path)

    static let ngramSource = """
        const uint head = thread_position_in_grid.x;
        const uint tNew = thread_position_in_grid.y;
        const uint b = thread_position_in_grid.z;
        if (head >= (uint)HEADS || tNew >= (uint)NEW) return;
        const uint HLEN = CTX + NEW;
        const device long* hist = history + (size_t)b * (size_t)HLEN;
        const long eos = EOS;
        int previous[16];
        int last = -1;
        for (uint t = 0; t < HLEN; ++t) {
            previous[t] = last;
            if (hist[t] == eos) last = (int)t;
        }
        auto shifted = [&](int sh, int t) -> long {
            if (sh == 0) return hist[t];
            const int inSegment = t - (previous[t] + 1);
            const int source = t - sh;
            return (inSegment >= sh && source >= 0) ? hist[source] : eos;
        };
        const int t = (int)(CTX + tNew);
        const int ngram = head < (uint)HPN ? 2 : 3;
        long mixed = shifted(0, t) * multipliers[0];
        for (int p = 1; p < ngram; ++p) {
            mixed ^= shifted(p, t) * multipliers[p];
        }
        long size = sizes[head];
        long r = mixed % size;
        if (r != 0 && ((r < 0) != (size < 0))) r += size;
        out[((size_t)b * (size_t)NEW + (size_t)tNew) * (size_t)HEADS + head] =
            (int)(r + offsets[head]);
        """

    nonisolated(unsafe) static let ngramKernel = MLXFast.metalKernel(
        name: "track_ple_chain_ngram_ids",
        inputNames: ["history", "multipliers", "sizes", "offsets"],
        outputNames: ["out"],
        source: ngramSource, header: "", ensureRowContiguous: true)

    struct NGramHash {
        let multipliers: [Int64]
        let sizes: [Int64]
        let offsets: [Int64]
        let ngramSize: Int
        let headsPerNGram: Int
        let ngramHeads: Int
        let eosTokenId: Int
    }

    static func ngramHash(cfg: Qwen4ExpTextConfiguration, pleLayerIndex: Int) -> NGramHash {
        let ngramHeads = (cfg.ngramSize - 1) * cfg.headsPerNGram
        var sizes: [Int64] = []
        var offsets: [Int64] = []
        var total: Int64 = 0
        for head in 0 ..< ngramHeads {
            let global = pleLayerIndex * ngramHeads + head
            let size = Int64(nthPrimeAfter(cfg.ngramVocabSizeBase - 1, count: global + 1))
            sizes.append(size)
            offsets.append(total)
            total += size
        }
        let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
        let maxLong = UInt64(Int64.max)
        let half = Swift.max(UInt64(1), (maxLong / UInt64(Swift.max(cfg.vocabularySize, 1))) / 2)
        let baseSeed = UInt64(bitPattern: Int64(cfg.seed)) &+ (10007 &* UInt64(pleLayerIndex))
        var multipliers: [Int64] = []
        for i in 0 ..< cfg.ngramSize {
            let mixed = splitmix64(baseSeed &+ (gamma &* UInt64(i + 1)))
            multipliers.append(Int64(2 &* (mixed % half) &+ 1))
        }
        return NGramHash(
            multipliers: multipliers, sizes: sizes, offsets: offsets,
            ngramSize: cfg.ngramSize, headsPerNGram: cfg.headsPerNGram,
            ngramHeads: ngramHeads, eosTokenId: cfg.eosTokenId)
    }

    /// Host integers, same contract as `Qwen4ExpNGramEmbedding.hostRowIds`.
    static func hostRowIds(_ hash: NGramHash, history: [[Int64]], newCount: Int) -> [Int] {
        let eos = Int64(hash.eosTokenId)
        var out: [Int] = []
        out.reserveCapacity(history.count * newCount * hash.ngramHeads)
        for row in history {
            let T = row.count
            var previous = [Int](repeating: -1, count: T)
            var last = -1
            for t in 0 ..< T {
                previous[t] = last
                if row[t] == eos { last = t }
            }
            func shifted(_ s: Int, _ t: Int) -> Int64 {
                if s == 0 { return row[t] }
                let inSegment = t - (previous[t] + 1)
                let source = t - s
                return (inSegment >= s && source >= 0) ? row[source] : eos
            }
            for t in Swift.max(0, T - newCount) ..< T {
                for ngram in 2 ... hash.ngramSize {
                    var mixed = shifted(0, t) &* hash.multipliers[0]
                    for p in 1 ..< ngram { mixed ^= shifted(p, t) &* hash.multipliers[p] }
                    let low = (ngram - 2) * hash.headsPerNGram
                    for head in low ..< low + hash.headsPerNGram {
                        var r = mixed % hash.sizes[head]
                        if r != 0, (r < 0) != (hash.sizes[head] < 0) { r += hash.sizes[head] }
                        out.append(Int(r + hash.offsets[head]))
                    }
                }
            }
        }
        return out
    }

    static func fusedRowIds(
        ids: MLXArray, previousContext: MLXArray, hash: NGramHash
    ) -> MLXArray? {
        let B = ids.dim(0), S = ids.dim(1), ctx = previousContext.dim(1)
        guard
            B >= 1, S >= 1, S <= 8, ctx >= 1, ctx + S <= 16,
            hash.ngramSize == 3, hash.headsPerNGram == 8, hash.ngramHeads == 16,
            ids.dtype == .int32 || ids.dtype == .int64,
            previousContext.dtype == .int32 || previousContext.dtype == .int64
        else { return nil }
        let history = concatenated([previousContext, ids], axis: 1).asType(.int64)
        let mul = MLXArray(hash.multipliers)
        let sizes = MLXArray(hash.sizes)
        let offsets = MLXArray(hash.offsets)
        return ngramKernel(
            [history, mul, sizes, offsets],
            template: [
                ("HEADS", hash.ngramHeads), ("HPN", hash.headsPerNGram),
                ("NEW", S), ("CTX", ctx), ("EOS", hash.eosTokenId),
            ],
            grid: (hash.ngramHeads, S, B), threadGroup: (hash.ngramHeads, 1, 1),
            outputShapes: [[B, S, hash.ngramHeads]], outputDTypes: [.int32])[0]
    }

    // MARK: - CPU mirror (f32 sequential; fused vs staged must match bits)

    static func roundTo(_ x: Float, dtype: DType) -> Float {
        switch dtype {
        case .float32: return x
        case .float16:
            return Float(Float16(x))
        case .bfloat16:
            var u = x.bitPattern
            let lsb = (u >> 16) & 1
            u = u &+ (0x7FFF &+ lsb)
            u &= 0xFFFF_0000
            return Float(bitPattern: u)
        default: return x
        }
    }

    static func cpuRmsInv(_ x: [Float], eps: Float) -> Float {
        var acc: Float = 0
        for v in x { acc += v * v }
        return 1.0 / sqrt(acc / Float(x.count) + eps)
    }

    static func cpuGate(_ dot: Float, hidden: Int, dtype: DType) -> Float {
        let divisor = roundTo(sqrt(Float(hidden)), dtype: dtype)
        let floor = roundTo(1e-6, dtype: dtype)
        var g = roundTo(dot / divisor, dtype: dtype)
        let ax = abs(g)
        let mag = ax > floor ? ax : (ax.isNaN ? ax : floor)
        g = roundTo(sqrt(mag), dtype: dtype) * (g > 0 ? 1 : (g < 0 ? -1 : 0))
        let y = 1 / (1 + exp(abs(g)))
        return g < 0 ? y : 1 - y
    }

    static func cpuUnfusedPrepare(
        key: [Float], query: [Float], value: [Float],
        kScale: [Float], qScale: [Float], cScale: [Float],
        hidden: Int, hcCount: Int, S: Int, eps: Float, dtype: DType
    ) -> (gated: [Float], normed: [Float]) {
        let W = hcCount * hidden
        var gated = [Float](repeating: 0, count: S * W)
        var normed = [Float](repeating: 0, count: S * W)
        for s in 0 ..< S {
            for hc in 0 ..< hcCount {
                let kOff = s * W + hc * hidden
                let k = Array(key[kOff ..< (kOff + hidden)])
                let q = Array(query[kOff ..< (kOff + hidden)])
                let kinv = cpuRmsInv(k, eps: eps)
                let qinv = cpuRmsInv(q, eps: eps)
                var prod: [Float] = []
                prod.reserveCapacity(hidden)
                for d in 0 ..< hidden {
                    let kn = roundTo(roundTo(k[d] * kinv, dtype: dtype) * kScale[hc * hidden + d], dtype: dtype)
                    let qn = roundTo(roundTo(q[d] * qinv, dtype: dtype) * qScale[hc * hidden + d], dtype: dtype)
                    prod.append(roundTo(kn * qn, dtype: dtype))
                }
                var dot: Float = 0
                for p in prod { dot = roundTo(p + dot, dtype: dtype) }
                let sgm = cpuGate(dot, hidden: hidden, dtype: dtype)
                var gvec: [Float] = []
                gvec.reserveCapacity(hidden)
                for d in 0 ..< hidden {
                    let v = roundTo(sgm * value[s * hidden + d], dtype: dtype)
                    gated[kOff + d] = v
                    gvec.append(v)
                }
                let inv = cpuRmsInv(gvec, eps: eps)
                for d in 0 ..< hidden {
                    normed[kOff + d] = roundTo(
                        roundTo(gvec[d] * inv, dtype: dtype) * cScale[hc * hidden + d], dtype: dtype)
                }
            }
        }
        return (gated, normed)
    }

    static func cpuFusedPrepare(
        key: [Float], query: [Float], value: [Float],
        kScale: [Float], qScale: [Float], cScale: [Float],
        hidden: Int, hcCount: Int, S: Int, eps: Float, dtype: DType
    ) -> (gated: [Float], normed: [Float]) {
        let W = hcCount * hidden
        var gated = [Float](repeating: 0, count: S * W)
        var normed = [Float](repeating: 0, count: S * W)
        for s in 0 ..< S {
            let vOff = s * hidden
            for hc in 0 ..< hcCount {
                let off = s * W + hc * hidden
                var ksq: Float = 0
                var qsq: Float = 0
                for d in 0 ..< hidden {
                    ksq += key[off + d] * key[off + d]
                    qsq += query[off + d] * query[off + d]
                }
                let kinv = 1.0 / sqrt(ksq / Float(hidden) + eps)
                let qinv = 1.0 / sqrt(qsq / Float(hidden) + eps)
                var dot: Float = 0
                for d in 0 ..< hidden {
                    let kn = roundTo(
                        roundTo(key[off + d] * kinv, dtype: dtype) * kScale[hc * hidden + d],
                        dtype: dtype)
                    let qn = roundTo(
                        roundTo(query[off + d] * qinv, dtype: dtype) * qScale[hc * hidden + d],
                        dtype: dtype)
                    dot = roundTo(roundTo(kn * qn, dtype: dtype) + dot, dtype: dtype)
                }
                let sgm = cpuGate(dot, hidden: hidden, dtype: dtype)
                var gsq: Float = 0
                for d in 0 ..< hidden {
                    let v = roundTo(sgm * value[vOff + d], dtype: dtype)
                    gated[off + d] = v
                    gsq += v * v
                }
                let inv = 1.0 / sqrt(gsq / Float(hidden) + eps)
                for d in 0 ..< hidden {
                    normed[off + d] = roundTo(
                        roundTo(gated[off + d] * inv, dtype: dtype) * cScale[hc * hidden + d],
                        dtype: dtype)
                }
            }
        }
        return (gated, normed)
    }

    static func cpuConv(
        full: [Float], convW: [Float], gated: [Float],
        S: Int, W: Int, kc: Int, dilation: Int, dtype: DType
    ) -> [Float] {
        var out = [Float](repeating: 0, count: S * W)
        for t in 0 ..< S {
            for c in 0 ..< W {
                var acc: Float = 0
                for j in 0 ..< kc {
                    acc += full[(t + j * dilation) * W + c] * convW[c * kc + j]
                }
                let conv = roundTo(acc, dtype: dtype)
                let silu = roundTo(conv * cpuGateRaw(conv), dtype: dtype)
                out[t * W + c] = roundTo(gated[t * W + c] + silu, dtype: dtype)
            }
        }
        return out
    }

    /// Sigmoid in the array dtype, matching `mlx_sigmoid`.
    static func cpuGateRaw(_ g: Float) -> Float {
        let y = 1 / (1 + exp(abs(g)))
        return g < 0 ? y : 1 - y
    }

    static func splitmix64(_ value: UInt64) -> UInt64 {
        var v = value &+ 0x9E37_79B9_7F4A_7C15
        v = (v ^ (v >> 30)) &* 0xBF58_476D_1CE4_E5B9
        v = (v ^ (v >> 27)) &* 0x94D0_49BB_1331_11EB
        return v ^ (v >> 31)
    }

    static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var d = 3
        while d * d <= value {
            if value % d == 0 { return false }
            d += 2
        }
        return true
    }

    static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var p = start
        for _ in 0 ..< count {
            p += 1
            while !isPrime(p) { p += 1 }
        }
        return p
    }
}
