// MLXFAST-PLEGFULL: S>1 PLE gated+concat fusion. The gated kernel's arithmetic
// is verbatim from track_ple_gated; its normed output lands directly in the
// carried-state window `full` and trailing threadgroups copy the convState
// rows, so the separate `concatenated` launch and the normed round-trip are
// gone. Bit-exact: same gate chain, same rsqrt fold, same InT rounding.
import Foundation
import MLX

enum TrackPLEGatedFull {
    /// g0 [B,S,HC,1], value [B,S,H], cscale [W], divisor/floor 0-dim, eps,
    /// convState [B,N,W] -> gated [B,S,W], full [B,N+S,W]
    /// grid (H/4, HC, B*S + N), threadgroup (H/4, 1, 1): the first B*S
    /// threadgroups per hc run the gated arithmetic and write normed into
    /// full rows N..N+S-1; the last N copy convState rows 0..N-1.
    static let source = """
        constexpr int N_READS = 4;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hc = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float sums[32];

        if (row >= (uint)S) {
            const uint cr = row - (uint)S;
            const uint cbase = cr * W + hc * H;
            for (int i = 0; i < N_READS; ++i) {
                const uint d = lid * N_READS + i;
                full[cbase + d] = convState[cbase + d];
            }
            return;
        }
        const uint base = row * W + hc * H;

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
        const uint nbase = ((uint)N + row) * W + hc * H;
        for (int i = 0; i < N_READS; ++i) {
            const uint d = lid * N_READS + i;
            full[nbase + d] = static_cast<InT>(gx[i] * inv) * cscale[hc * H + d];
        }
        """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_ple_gated_full",
        inputNames: ["g0", "value", "cscale", "divisor", "floorv", "eps", "convState"],
        outputNames: ["gated", "full"],
        source: source,
        header: TrackFastPLEKernels.header + """
            template <typename T> METAL_FUNC T mlx_abs_t(T x) { return metal::abs(x); }
            template <typename T> METAL_FUNC T mlx_sqrt_t(T x) { return metal::precise::sqrt(x); }
            """,
        ensureRowContiguous: true)

    static func apply(
        g0: MLXArray, value: MLXArray, cScale: MLXArray, divisor: MLXArray,
        floor: MLXArray, convState: MLXArray, hcCount: Int, hidden: Int, eps: Float
    ) -> (gated: MLXArray, full: MLXArray)? {
        let B = value.dim(0), S = value.dim(1), W = hcCount * hidden
        let N = convState.dim(1)
        guard B == 1, hidden % 4 == 0, hidden / 4 <= 1024, value.dim(2) == hidden,
            convState.ndim == 3, convState.dim(0) == B, convState.dim(2) == W,
            convState.dtype == value.dtype, N >= 1
        else { return nil }
        let outs = kernel(
            [g0, value, cScale, divisor, floor, MLXArray(eps), convState],
            template: [
                ("InT", value.dtype), ("H", hidden), ("W", W), ("HC", hcCount),
                ("S", S), ("N", N),
            ],
            grid: (hidden / 4, hcCount, B * S + N), threadGroup: (hidden / 4, 1, 1),
            outputShapes: [[B, S, W], [B, N + S, W]],
            outputDTypes: [value.dtype, value.dtype])
        return (outs[0], outs[1])
    }
}
