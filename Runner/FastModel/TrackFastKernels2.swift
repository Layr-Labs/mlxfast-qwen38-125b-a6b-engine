// TrackFastKernels2.swift -- the elementwise/reduction fusions of the fast
// forward pass, BIT-EXACT with the engine's op chain.
//
// Every kernel reproduces the arithmetic of the MLX ops it replaces, not just
// their math: the `Sigmoid` functor evaluated in the array dtype (bf16, one
// rounding per operation, `1/(1+exp(|x|))` and its mirror), compiled `silu`
// as `x * sigmoid(x)` in bf16, `rms_single_row`'s layout (four consecutive
// elements per thread, a simd sum, then a simd sum over the per-simdgroup
// partials) with `precise::rsqrt`, the weight applied AFTER the bf16 rounding
// of `x * inv_mean`, the composed partial rope as bf16 multiplies and an add
// over bf16 cos/sin tables, and float accumulation in row order for the small
// reductions. The header below carries the shared helpers.

import Foundation
import MLX

extension TrackFastKernels {

    /// Shared helpers: MLX's `Sigmoid` / `LogAddExp` functors verbatim, and
    /// compiled `silu`.
    static let exactHeader = """
        template <typename T>
        METAL_FUNC T mlx_sigmoid(T x) {
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
        }
        template <typename T>
        METAL_FUNC T mlx_silu(T x) {
            T s = mlx_sigmoid(x);
            return x * s;
        }
        // MLX `col_reduce_small` over K rows (K < 32): threadgroup_y = min(8, K)
        // threads each fold rows y, y+ty, ... in order into an init of 0, then
        // thread 0 folds the partials in thread order.
        template <int K>
        METAL_FUNC float mlx_colsum_small_f32(thread const float* r) {
            constexpr int TY = K < 8 ? K : 8;
            float t[TY];
            for (int y = 0; y < TY; ++y) {
                float acc = 0.0f;
                for (int rr = y; rr < K; rr += TY) { acc = r[rr] + acc; }
                t[y] = acc;
            }
            float total = t[0];
            for (int j = 1; j < TY; ++j) { total = t[j] + total; }
            return total;
        }
        template <typename T>
        METAL_FUNC T mlx_logaddexp0(T x) {
            T y = 0;
            if (metal::isnan(x)) { return metal::numeric_limits<T>::quiet_NaN(); }
            constexpr T inf = metal::numeric_limits<T>::infinity();
            T maxval = metal::max(x, y);
            T minval = metal::min(x, y);
            return (minval == -inf || maxval == inf)
                ? maxval
                : (maxval + log1p(metal::exp(minval - maxval)));
        }
        """

    // MARK: inject + group RMS norm  (rms_single_row layout: 4 elements/thread)

    /// stream = residual + out ⊗ inject (bf16), normed = rms(stream_group) * scale.
    ///   residual [B,S,W] (or [B,S,H] when TILE), out [B,S,H], inject [B,S,HC], scale [W], eps
    ///   -> stream [B,S,W], normed [B,S,W]
    /// grid (H/4, HC, B*S), threadgroup (H/4, 1, 1)   (H = 2560 -> 640 threads, 20 simdgroups)
    static let injectNormSource = """
        constexpr int N_READS = 4;
        constexpr uint NT = H / N_READS;
        const uint row = thread_position_in_grid.z;
        const uint hc = thread_position_in_grid.y;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float local_sums[32];
        const uint base = row * W + hc * H;
        float inj = 0.0f;
        InT inj_t = InT(0);
        if (HAS_INJECT) { inj_t = inject[row * HC + hc]; inj = static_cast<float>(inj_t); }
        (void)inj;
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const uint src = TILE ? (row * H + d) : (base + d);
            InT r = residual[src];
            if (HAS_INJECT) {
                InT sp = out[row * H + d] * inj_t;
                r = r + sp;
            }
            stream[base + d] = r;
            thread_x[i] = static_cast<float>(r);
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { local_sums[lane] = 0; }
        if (lane == 0) { local_sums[sg] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(local_sums[lane]);
        const float inv_mean = metal::precise::rsqrt(acc / (float)H + eps);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            InT n = static_cast<InT>(thread_x[i] * inv_mean);
            normed[base + d] = n * scale[hc * H + d];
        }
        """

    nonisolated(unsafe) static let injectNormKernel = MLXFast.metalKernel(
        name: "track_inject_norm",
        inputNames: ["residual", "out", "inject", "scale", "eps"],
        outputNames: ["stream", "normed"],
        source: injectNormSource, header: exactHeader, ensureRowContiguous: true)

    static func injectNorm(
        residual: MLXArray, out: MLXArray?, inject: MLXArray?, scale: MLXArray,
        hcCount: Int, hidden: Int, eps: Float, tile: Bool
    ) -> (stream: MLXArray, normed: MLXArray) {
        let B = residual.dim(0), S = residual.dim(1)
        let W = hcCount * hidden
        precondition(hidden % 4 == 0 && hidden / 4 <= 1024)
        let hasInject = out != nil
        let outs = injectNormKernel(
            [residual, out ?? residual, inject ?? residual, scale, MLXArray(eps)],
            template: [
                ("InT", residual.dtype), ("H", hidden), ("W", W), ("HC", hcCount),
                ("HAS_INJECT", hasInject), ("TILE", tile),
            ],
            grid: (hidden / 4, hcCount, B * S), threadGroup: (hidden / 4, 1, 1),
            outputShapes: [[B, S, W], [B, S, W]],
            outputDTypes: [residual.dtype, residual.dtype])
        return (outs[0], outs[1])
    }

    // MARK: mixer combine

    /// input[d] = bf16( sum_s f32(bf16(sigmoid(w[s,d])) * normed[s,d]) ) (row-order f32 sum, one rounding)
    /// inject[s] = 2 * sigmoid(lo[LW - HC + s])   (bf16 ops)
    static let mixSource = """
        const uint d = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (d >= H) return;
        // `sum` over the stream axis of a bf16 array accumulates in bf16, one
        // rounding per add, in row order.
        InT acc = InT(0);
        for (int s = 0; s < HC; ++s) {
            const uint i = row * W + s * H + d;
            InT sg = mlx_sigmoid(w[i]);
            InT p = sg * normed[i];
            acc = acc + p;
        }
        input[row * H + d] = acc;
        if (HAS_INJECT && d < HC) {
            InT x = lo[row * LW + (LW - HC) + d];
            InT sg = mlx_sigmoid(x);
            inject[row * HC + d] = InT(2) * sg;
        }
        """

    nonisolated(unsafe) static let mixKernel = MLXFast.metalKernel(
        name: "track_hc_mix",
        inputNames: ["w", "normed", "lo"],
        outputNames: ["input", "inject"],
        source: mixSource, header: exactHeader, ensureRowContiguous: true)

    static func hcMix(
        w: MLXArray, normed: MLXArray, lo: MLXArray, hcCount: Int, hidden: Int, hasInject: Bool
    ) -> (input: MLXArray, inject: MLXArray) {
        let B = w.dim(0), S = w.dim(1)
        let outs = mixKernel(
            [w, normed, lo],
            template: [
                ("InT", w.dtype), ("H", hidden), ("W", hcCount * hidden), ("HC", hcCount),
                ("LW", lo.dim(2)), ("HAS_INJECT", hasInject),
            ],
            grid: (hidden, B * S, 1), threadGroup: (256, 1, 1),
            outputShapes: [[B, S, hidden], [B, S, hcCount]],
            outputDTypes: [w.dtype, w.dtype])
        return (outs[0], outs[1])
    }

    // MARK: compiled silu over the leading columns

    static let siluHeadSource = """
        const uint j = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (j >= LMIX) return;
        out[row * LMIX + j] = mlx_silu(lo[row * LW + j]);
        """

    nonisolated(unsafe) static let siluHeadKernel = MLXFast.metalKernel(
        name: "track_silu_head",
        inputNames: ["lo"],
        outputNames: ["out"],
        source: siluHeadSource, header: exactHeader, ensureRowContiguous: true)

    static func siluHead(lo: MLXArray, width: Int) -> MLXArray {
        let B = lo.dim(0), S = lo.dim(1)
        return siluHeadKernel(
            [lo], template: [("InT", lo.dtype), ("LW", lo.dim(2)), ("LMIX", width)],
            grid: (width, B * S, 1), threadGroup: (256, 1, 1),
            outputShapes: [[B, S, width]], outputDTypes: [lo.dtype])[0]
    }

    // MARK: gated output RMS norm (deltanet): rms over Dv=128 (32 threads x 4), weight after rounding,
    //       then f32 sigmoid(z) * f32(out) -> bf16

    /// y [B,S,Hv,Dv], proj [B,S,PW] (z at Z_OFF), w [Dv], eps -> out [B,S,Hv*Dv]
    /// grid (32, Hv, B*S), threadgroup (32,1,1)
    static let gatedRMSSource = """
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hv = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        threadgroup float local_sums[32];
        const uint ybase = (row * Hv + hv) * Dv;
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            thread_x[i] = static_cast<float>(y[ybase + lid * N_READS + i]);
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        if (lane >= 1) { local_sums[lane] = 0; }
        if (lane == 0) { local_sums[0] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(local_sums[lane]);
        const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + eps);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            InT n = w[d] * static_cast<InT>(thread_x[i] * inv_mean);
            const float z = static_cast<float>(proj[row * PW + Z_OFF + hv * Dv + d]);
            const float g = mlx_sigmoid(z);
            out[ybase + d] = static_cast<InT>(g * static_cast<float>(n));
        }
        """

    nonisolated(unsafe) static let gatedRMSKernel = MLXFast.metalKernel(
        name: "track_gated_rms",
        inputNames: ["y", "proj", "w", "eps"],
        outputNames: ["out"],
        source: gatedRMSSource, header: exactHeader, ensureRowContiguous: true)

    static func gatedRMS(y: MLXArray, proj: MLXArray, w: MLXArray, zOffset: Int, eps: Float)
        -> MLXArray
    {
        let B = y.dim(0), S = y.dim(1), Hv = y.dim(2), Dv = y.dim(3)
        precondition(Dv == 128)
        return gatedRMSKernel(
            [y, proj, w, MLXArray(eps)],
            template: [
                ("InT", y.dtype), ("Hv", Hv), ("Dv", Dv), ("PW", proj.dim(2)),
                ("Z_OFF", zOffset),
            ],
            grid: (32, Hv, B * S), threadGroup: (32, 1, 1),
            outputShapes: [[B, S, Hv * Dv]], outputDTypes: [y.dtype])[0]
    }

    // MARK: attention prep: q/k RMS norm (64 threads x 4) + composed partial rope over bf16 tables

    /// qkv [B,S,QW] rows: q(HQ*D) | gate(HQ*D) | k(HK*D) | v(HK*D) | ...
    /// cos, sin [S, ROT] bf16 (the reference's tables cast to the activation dtype)
    ///   -> q [B,HQ,S,D], k [B,HK,S,D], v [B,HK,S,D]
    /// grid (D/4, HQ + 2*HK, B*S), threadgroup (D/4, 1, 1)   (D = 256 -> 64 threads, 2 simdgroups)
    static let attnPrepSource = """
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint h = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint b = row / S;
        const uint s = row % S;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float local_sums[32];
        threadgroup InT vec[D];

        const bool isQ = h < HQ;
        const bool isV = h >= HQ + HK;
        uint hh, src;
        if (isQ) { hh = h; src = row * QW + h * D; }
        else if (!isV) { hh = h - HQ; src = row * QW + 2 * HQ * D + hh * D; }
        else { hh = h - HQ - HK; src = row * QW + 2 * HQ * D + HK * D + hh * D; }
        if (isV) {
            for (int i = 0; i < N_READS; ++i) {
                const uint d = lid * N_READS + i;
                vout[((b * HK + hh) * S + s) * D + d] = qkv[src + d];
            }
            return;
        }
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            thread_x[i] = static_cast<float>(qkv[src + lid * N_READS + i]);
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        constexpr uint simd_groups = (D + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { local_sums[lane] = 0; }
        if (lane == 0) { local_sums[sg] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(local_sums[lane]);
        const float inv_mean = metal::precise::rsqrt(acc / (float)D + eps);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT wgt = isQ ? qnorm[d] : knorm[d];
            vec[d] = wgt * static_cast<InT>(thread_x[i] * inv_mean);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr int hrot = ROT / 2;
        device InT* dst = isQ ? (qout + ((b * HQ + hh) * S + s) * D) : (kout + ((b * HK + hh) * S + s) * D);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            InT o = vec[d];
            if (d < ROT) {
                const InT c = cosb[s * ROT + d];
                const InT sn = sinb[s * ROT + d];
                if (d < hrot) {
                    InT x1 = vec[d];
                    InT x2 = vec[d + hrot];
                    InT t1 = x1 * c;
                    InT t2 = (-x2) * sn;
                    o = t1 + t2;
                } else {
                    InT x2 = vec[d];
                    InT x1 = vec[d - hrot];
                    InT t1 = x2 * c;
                    InT t2 = x1 * sn;
                    o = t1 + t2;
                }
            }
            dst[d] = o;
        }
        """

    nonisolated(unsafe) static let attnPrepKernel = MLXFast.metalKernel(
        name: "track_attn_prep",
        inputNames: ["qkv", "qnorm", "knorm", "cosb", "sinb", "eps"],
        outputNames: ["qout", "kout", "vout"],
        source: attnPrepSource, header: exactHeader, ensureRowContiguous: true)

    static func attnPrep(
        qkv: MLXArray, qNorm: MLXArray, kNorm: MLXArray, cos: MLXArray, sin: MLXArray,
        heads: Int, kvHeads: Int, headDim: Int, rotaryDims: Int, eps: Float
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let B = qkv.dim(0), S = qkv.dim(1)
        precondition(headDim % 4 == 0 && rotaryDims % 8 == 0 && cos.dim(1) == rotaryDims)
        let outs = attnPrepKernel(
            [qkv, qNorm, kNorm, cos, sin, MLXArray(eps)],
            template: [
                ("InT", qkv.dtype), ("D", headDim), ("HQ", heads), ("HK", kvHeads), ("S", S),
                ("QW", qkv.dim(2)), ("ROT", rotaryDims),
            ],
            grid: (headDim / 4, heads + 2 * kvHeads, B * S), threadGroup: (headDim / 4, 1, 1),
            outputShapes: [[B, heads, S, headDim], [B, kvHeads, S, headDim], [B, kvHeads, S, headDim]],
            outputDTypes: [qkv.dtype, qkv.dtype, qkv.dtype])
        return (outs[0], outs[1], outs[2])
    }

    // MARK: attention output gate

    /// out[b,s,h*D+d] = att[b,h,s,d] * sigmoid(gate[b,s,h*D+d])   (bf16 ops)
    static let attnGateSource = """
        const uint j = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (j >= HQ * D) return;
        const uint h = j / D;
        const uint d = j % D;
        const uint b = row / S;
        const uint s = row % S;
        const InT a = att[((b * HQ + h) * S + s) * D + d];
        const InT g = qkv[row * QW + GATE_OFF + j];
        out[row * HQ * D + j] = a * mlx_sigmoid(g);
        """

    nonisolated(unsafe) static let attnGateKernel = MLXFast.metalKernel(
        name: "track_attn_gate",
        inputNames: ["att", "qkv"],
        outputNames: ["out"],
        source: attnGateSource, header: exactHeader, ensureRowContiguous: true)

    static func attnGate(att: MLXArray, qkv: MLXArray, gateOffset: Int) -> MLXArray {
        let B = att.dim(0), HQ = att.dim(1), S = att.dim(2), D = att.dim(3)
        return attnGateKernel(
            [att, qkv],
            template: [
                ("InT", att.dtype), ("HQ", HQ), ("D", D), ("S", S), ("QW", qkv.dim(2)),
                ("GATE_OFF", gateOffset),
            ],
            grid: (HQ * D, B * S, 1), threadGroup: (256, 1, 1),
            outputShapes: [[B, S, HQ * D]], outputDTypes: [att.dtype])[0]
    }

    // MARK: MoE combine

    /// out = bf16(sum_k f32(routed[k]) * w[k]) + sigmoid(gate) * shared   (bf16 ops after the f32 sum)
    static let moeCombineSource = """
        const uint d = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (d >= H) return;
        float prod[K];
        for (int k = 0; k < K; ++k) {
            prod[k] = static_cast<float>(routed[(row * K + k) * H + d]) * w[row * K + k];
        }
        const InT r = static_cast<InT>(mlx_colsum_small_f32<K>(prod));
        const InT sg = mlx_sigmoid(gate[row]);
        const InT sh = sg * shared[row * H + d];
        out[row * H + d] = r + sh;
        """

    nonisolated(unsafe) static let moeCombineKernel = MLXFast.metalKernel(
        name: "track_moe_combine",
        inputNames: ["routed", "w", "shared", "gate"],
        outputNames: ["out"],
        source: moeCombineSource, header: exactHeader, ensureRowContiguous: true)

    static func moeCombine(routed: MLXArray, w: MLXArray, shared: MLXArray, gate: MLXArray)
        -> MLXArray
    {
        let B = routed.dim(0), S = routed.dim(1), K = routed.dim(2), H = routed.dim(3)
        precondition(w.dtype == .float32)
        return moeCombineKernel(
            [routed, w, shared, gate],
            template: [("InT", routed.dtype), ("K", K), ("H", H)],
            grid: (H, B * S, 1), threadGroup: (256, 1, 1),
            outputShapes: [[B, S, H]], outputDTypes: [routed.dtype])[0]
    }

    // MARK: SwiGLU over a fused gate|up output (compiled silu, then bf16 multiply)

    static let swigluSource = """
        const uint f = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (f >= F) return;
        const InT g = gu[row * 2 * F + f];
        const InT u = gu[row * 2 * F + F + f];
        out[row * F + f] = mlx_silu(g) * u;
        """

    nonisolated(unsafe) static let swigluKernel = MLXFast.metalKernel(
        name: "track_swiglu",
        inputNames: ["gu"],
        outputNames: ["out"],
        source: swigluSource, header: exactHeader, ensureRowContiguous: true)

    static func swiglu(gu: MLXArray) -> MLXArray {
        let B = gu.dim(0), S = gu.dim(1), F = gu.dim(2) / 2
        return swigluKernel(
            [gu], template: [("InT", gu.dtype), ("F", F)],
            grid: (F, B * S, 1), threadGroup: (256, 1, 1),
            outputShapes: [[B, S, F]], outputDTypes: [gu.dtype])[0]
    }
}
