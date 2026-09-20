import MLX

/// Large affine QMV decode projections: eight independent rows per SIMD group.
/// Keep the promoted qmv_fast per-row arithmetic while sharing input preparation
/// across twice as many outputs and dispatching half as many threadgroups.
enum TrackWideRowDecode {
    private static let kernel = MLXFast.metalKernel(
        name: "track_large_qmv_eight_rows",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: #"""
            constexpr int ROWS = 8;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int first = int(threadgroup_position_in_grid.y) * (2 * ROWS)
                + int(sg) * ROWS;
            float result[ROWS];
            qmv_fast_reg<T, 32, 4, ROWS>(
                w, scales, biases, x, K, first, lane, result);
            // Each lane already holds every post-simd_sum result. Static
            // indexing avoids a dynamically indexed register-array gather.
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (lane == uint(r)) { y[first + r] = static_cast<T>(result[r]); }
            }
            """#,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
            + TrackFastMoEKernels.regHelpers,
        ensureRowContiguous: true)

    static func apply(_ q: TrackQuantWeight, _ x: MLXArray) -> MLXArray? {
        guard x.ndim >= 2, x.dtype == .bfloat16,
            q.mode == .affine, q.bits == 4, q.groupSize == 32,
            q.weight.ndim == 2, q.weight.dtype == .uint32,
            let bias = q.biases, bias.dtype == .bfloat16,
            q.scales.dtype == .bfloat16,
            StreamOrDevice.default.stream === Stream.gpu
        else { return nil }
        let k = x.dim(-1), n = q.rows
        guard x.size == k, k >= 512, k % 512 == 0, n >= 1024, n % 16 == 0,
            q.weight.shape == [n, k / 8], q.scales.shape == [n, k / 32],
            bias.shape == q.scales.shape
        else { return nil }
        var shape = x.shape
        shape[shape.count - 1] = n
        return kernel(
            [x, q.weight, q.scales, bias],
            template: [("T", x.dtype), ("K", k)],
            grid: (32, (n / 16) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [shape], outputDTypes: [x.dtype])[0]
    }
}
