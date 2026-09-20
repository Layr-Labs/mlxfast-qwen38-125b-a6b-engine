// TrackFastPLE.swift -- the layer-2 PLE block as three launches.
//
// WHAT THE REFERENCE DISPATCHES, per forward, after the n-gram rows arrive
// (the row gather stays exactly where it is: it is host and IO work):
//
//   key_proj GEMV, norm_key (rms + scale), value_proj GEMV,
//   norm_query (rms + scale), key*query, sum over the last axis,
//   / sqrt(hidden), abs, maximum, sqrt, sign, multiply, sigmoid,
//   multiply by value, norm_conv (rms + scale), concat with the carried
//   state, a DILATED depthwise conv1d, silu, and the residual add.
//
// That is ~22 launches over [1, S, 10240] arrays, and the conv is the worst
// of them: `conv_1D_gpu` takes the fully separable fast path only when
// `wt_dilation[0] == 1`, and this convolution has dilation 3 (ngram_size),
// so it falls through to `implicit_gemm_conv_2D_gpu` with groups = 10240 --
// one threadgroup per group, each computing a tile of which a single row is
// wanted. Measured on this box, concat + conv + silu alone is ~80 us of a
// ~180-300 us block.
//
// Three kernels replace all of it except the two projections and the one
// reduction:
//
//   `track_ple_prod`  norm_key(key_proj) * norm_query(stream)
//   <MLX's own sum over the last axis, untouched>
//   `track_ple_gated` the gate transform, the gated value, norm_conv
//   `track_ple_conv`  the dilated conv, silu, and the residual add
//
// EXACTNESS. The two norms are `rms_single_row`'s layout at axis 2560 -- four
// consecutive elements per thread, a simd sum, a simd sum over the
// per-simdgroup partials, `precise::rsqrt`, and the weight applied AFTER the
// bf16 rounding of `x * inv_mean` -- the same layout `track_inject_norm`
// already carries. The scalar chain is MLX's own functors evaluated in the
// array dtype, one rounding per operation: `Divide` x / y, `Abs`
// `metal::abs`, `Maximum` `isnan(x) ? x : (x > y ? x : y)`, `Sqrt`
// `metal::precise::sqrt`, `Sign` `(x > 0) - (x < 0)`, `Sigmoid` and compiled
// `silu` as already used elsewhere in this tree. The two scalars the chain
// divides and clamps by ride as template bit patterns: `InT(float)` is the
// same round-to-nearest cast `asMLXArray(dtype:)` performs, so they carry
// the same bf16 rounding. The convolution accumulates its `kernel_size` taps
// in float in ascending tap order and rounds once, which is what the
// reference's implicit GEMM does for this shape; that is the one claim here
// that is empirical rather than structural, and it is what the comparison
// test checks element by element at S = 1..8.
//
// The reduction between `track_ple_prod` and `track_ple_gated` is folded
// inside `track_ple_prod` for S <= 7 -- the window sizes where MLX dispatches
// `row_reduce_looped`, whose per-thread four-element InT fold, simd_sum and
// cross-simdgroup partials the kernel reproduces exactly. At S >= 8 MLX
// switches to `row_reduce_simple`, whose fold differs, so wide windows still
// take MLX's own `sum` over the last axis.

import Foundation
import MLX

enum TrackFastPLEKernels {

    static let header = TrackFastKernels.exactHeader + """
        template <typename T>
        METAL_FUNC T mlx_maximum(T x, T y) {
            if (metal::isnan(x)) { return x; }
            return x > y ? x : y;
        }
        template <typename T>
        METAL_FUNC T mlx_sign(T x) {
            return static_cast<T>((x > T(0)) - (x < T(0)));
        }
        """

    // MARK: norm_key(key_proj(e)) * norm_query(stream)

    /// keyFlat [B,S,W], stream [B,S,W], kscale [W], qscale [W]; eps rides as a
    /// template bit pattern.
    ///   -> prod [B,S,W], dots [B,S,HC] (the row dot, folded in the same pass)
    /// grid (H/4, HC, B*S), threadgroup (H/4, 1, 1)
    static let prodSource = """
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hc = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float ksums[32];
        threadgroup float qsums[32];
        threadgroup InT dsums[32];
        const uint base = row * W + hc * H;
        const float eps = as_type<float>((uint)EPS_BITS);
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
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { ksums[lane] = 0; qsums[lane] = 0; }
        if (lane == 0) { ksums[sg] = kacc; qsums[sg] = qacc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kacc = simd_sum(ksums[lane]);
        qacc = simd_sum(qsums[lane]);
        const float kinv = metal::precise::rsqrt(kacc / (float)H + eps);
        const float qinv = metal::precise::rsqrt(qacc / (float)H + eps);
        // MLXFAST-PRODDOT: the row's dot product is folded in the same pass,
        // bit-exact with `row_reduce_looped` -- the kernel MLX dispatches for
        // this reduction whenever 4*S < 32 rows, i.e. exactly S <= 7. Same
        // thread-to-element mapping (lid * 4 + i), same sequential InT fold,
        // same simd_sum, same cross-simdgroup partials in the InT accumulator.
        InT dot = InT(0);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT kn = static_cast<InT>(kx[i] * kinv) * kscale[hc * H + d];
            const InT qn = static_cast<InT>(qx[i] * qinv) * qscale[hc * H + d];
            const InT p = kn * qn;
            prod[base + d] = p;
            dot = p + dot;
        }
        dot = simd_sum(dot);
        if (sg == 0 && lane >= simd_groups) { dsums[lane] = InT(0); }
        if (lane == 0) { dsums[sg] = dot; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = simd_sum(dsums[lane]);
        if (sg == 0 && lane == 0) { dots[row * HC + hc] = dot; }
        """

    nonisolated(unsafe) static let prodKernel = MLXFast.metalKernel(
        name: "track_ple_prod",
        inputNames: ["keyFlat", "stream", "kscale", "qscale"],
        outputNames: ["prod", "dots"],
        source: prodSource, header: header, ensureRowContiguous: true)

    static func prod(
        keyFlat: MLXArray, stream: MLXArray, kScale: MLXArray, qScale: MLXArray,
        hcCount: Int, hidden: Int, eps: Float
    ) -> (prod: MLXArray, dots: MLXArray) {
        let B = keyFlat.dim(0), S = keyFlat.dim(1), W = hcCount * hidden
        precondition(hidden % 4 == 0 && hidden / 4 <= 1024 && keyFlat.dim(2) == W)
        let outs = prodKernel(
            [keyFlat, stream, kScale, qScale],
            template: [("InT", keyFlat.dtype), ("H", hidden), ("W", W), ("HC", hcCount),
                       ("EPS_BITS", Int(eps.bitPattern))],
            grid: (hidden / 4, hcCount, B * S), threadGroup: (hidden / 4, 1, 1),
            outputShapes: [[B, S, W], [B, S, hcCount]],
            outputDTypes: [keyFlat.dtype, keyFlat.dtype])
        return (outs[0], outs[1])
    }

    // MARK: the gate scalar chain, the gated value, and norm_conv

    /// g0 [B,S,HC] (the reduced dot), value [B,S,H], cscale [W]; the divisor,
    /// floor and eps scalars ride as template bit patterns.
    ///   -> gated [B,S,W], normed [B,S,W]
    /// grid (H/4, HC, B*S), threadgroup (H/4, 1, 1)
    static let gatedSource = """
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hc = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float sums[32];
        const uint base = row * W + hc * H;
        const float eps = as_type<float>((uint)EPS_BITS);
        // The two scalars the reference's `/` and `maximum` build, carried as
        // bit patterns: `InT(float)` is the same round-to-nearest cast
        // `asMLXArray(dtype:)` performs, so the divide and the clamp see the
        // identical bf16 values without a per-window scalar upload.
        const InT divisor = InT(as_type<float>((uint)DIVISOR_BITS));
        const InT floorv = InT(as_type<float>((uint)FLOOR_BITS));

        InT g = g0[row * HC + hc] / divisor;
        g = mlx_sqrt_t(mlx_maximum(mlx_abs_t(g), floorv)) * mlx_sign(g);
        const InT sgm = mlx_sigmoid(g);

        float gx[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            const InT v = sgm * value[row * H + d];
            gated[base + d] = v;
            gx[i] = static_cast<float>(v);
            acc += gx[i] * gx[i];
        }
        acc = simd_sum(acc);
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        if (sg == 0 && lane >= simd_groups) { sums[lane] = 0; }
        if (lane == 0) { sums[sg] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = simd_sum(sums[lane]);
        const float inv = metal::precise::rsqrt(acc / (float)H + eps);
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            normed[base + d] = static_cast<InT>(gx[i] * inv) * cscale[hc * H + d];
        }
        """

    nonisolated(unsafe) static let gatedKernel = MLXFast.metalKernel(
        name: "track_ple_gated",
        inputNames: ["g0", "value", "cscale"],
        outputNames: ["gated", "normed"],
        source: gatedSource,
        header: header + """
            template <typename T> METAL_FUNC T mlx_abs_t(T x) { return metal::abs(x); }
            template <typename T> METAL_FUNC T mlx_sqrt_t(T x) { return metal::precise::sqrt(x); }
            """,
        ensureRowContiguous: true)

    static func gated(
        g0: MLXArray, value: MLXArray, cScale: MLXArray,
        hcCount: Int, hidden: Int, eps: Float
    ) -> (gated: MLXArray, normed: MLXArray) {
        let B = value.dim(0), S = value.dim(1), W = hcCount * hidden
        precondition(hidden % 4 == 0 && hidden / 4 <= 1024 && value.dim(2) == hidden)
        let outs = gatedKernel(
            [g0, value, cScale],
            template: [("InT", value.dtype), ("H", hidden), ("W", W), ("HC", hcCount),
                       ("EPS_BITS", Int(eps.bitPattern)),
                       ("DIVISOR_BITS", Int(Foundation.sqrt(Float(hidden)).bitPattern)),
                       ("FLOOR_BITS", Int(Float(1e-6).bitPattern))],
            grid: (hidden / 4, hcCount, B * S), threadGroup: (hidden / 4, 1, 1),
            outputShapes: [[B, S, W], [B, S, W]],
            outputDTypes: [value.dtype, value.dtype])
        return (outs[0], outs[1])
    }

    // MARK: the dilated depthwise convolution, silu, and the residual add

    /// full [B, N+S, W] (carried state ++ normed), convw [W, KC], gated [B,S,W]
    ///   -> out [B,S,W] = gated + silu(conv)
    /// grid (W, S, B), threadgroup (256, 1, 1)
    static let convSource = """
        const uint c = thread_position_in_grid.x;
        const uint t = thread_position_in_grid.y;
        const uint b = thread_position_in_grid.z;
        if (c >= (uint)W) return;
        const device InT* fb = full + (size_t)b * (size_t)(NIN) * (size_t)W;
        float acc = 0.0f;
        for (int j = 0; j < KC; ++j) {
            acc += static_cast<float>(fb[(size_t)(t + (uint)(j * DIL)) * (size_t)W + c])
                 * static_cast<float>(convw[c * KC + j]);
        }
        const size_t o = ((size_t)b * (size_t)S + (size_t)t) * (size_t)W + c;
        out[o] = gated[o] + mlx_silu(static_cast<InT>(acc));
        """

    nonisolated(unsafe) static let convKernel = MLXFast.metalKernel(
        name: "track_ple_conv",
        inputNames: ["full", "convw", "gated"],
        outputNames: ["out"],
        source: convSource, header: header, ensureRowContiguous: true)

    static func conv(
        full: MLXArray, convW: MLXArray, gated: MLXArray, dilation: Int
    ) -> MLXArray {
        let B = gated.dim(0), S = gated.dim(1), W = gated.dim(2)
        let kc = convW.dim(1)
        precondition(full.dim(2) == W && full.dim(1) == S + (kc - 1) * dilation)
        return convKernel(
            [full, convW, gated],
            template: [
                ("InT", gated.dtype), ("W", W), ("S", S), ("KC", kc), ("DIL", dilation),
                ("NIN", full.dim(1)),
            ],
            grid: (W, S, B), threadGroup: (256, 1, 1),
            outputShapes: [[B, S, W]], outputDTypes: [gated.dtype])[0]
    }
}
