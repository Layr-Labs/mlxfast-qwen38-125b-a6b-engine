import MLX

enum TrackLMHead {
    static let prepareSource = """
        const uint chunk = thread_position_in_grid.x;
        if (chunk >= K / 16) return;
        float values[16];
        sums[chunk] = load_vector<T, float, 16, 4>(x + chunk * 16, values);
        for (int i = 0; i < 16; ++i) {
            prepared[chunk * 16 + i] = values[i];
        }
        """

    static let projectSource = """
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.y * 8
            + simdgroup_index_in_threadgroup * 4;
        const device uint8_t* weights = (const device uint8_t*)w;
        float result[4] = {0};
        for (int k = 0; k < K; k += 512) {
            float values[16];
            for (int i = 0; i < 16; ++i) {
                values[i] = prepared[k + lane * 16 + i];
            }
            const float sum = sums[k / 16 + lane];
            for (int r = 0; r < 4; ++r) {
                const uint offset = (row + r) * (K / 32) + k / 32 + lane / 2;
                result[r] += qdot<float, 16, 4>(
                    weights + (row + r) * (K / 2) + k / 2 + lane * 8,
                    values, float(scales[offset]), float(biases[offset]), sum);
            }
        }
        for (int r = 0; r < 4; ++r) {
            const float value = simd_sum(result[r]);
            if (lane == 0) out[row + r] = T(value);
        }
        """

    nonisolated(unsafe) static let prepareKernel = MLXFast.metalKernel(
        name: "track_lm_head_prepare", inputNames: ["x"],
        outputNames: ["prepared", "sums"], source: prepareSource,
        header: TrackFastMoEKernels.helpersCore, ensureRowContiguous: true)

    nonisolated(unsafe) static let projectKernel = MLXFast.metalKernel(
        name: "track_lm_head_prepared", inputNames: ["w", "scales", "biases", "prepared", "sums"],
        outputNames: ["out"], source: projectSource,
        header: TrackFastMoEKernels.helpersCore, ensureRowContiguous: true)

    static func apply(_ x: MLXArray, weight q: TrackQuantWeight) -> MLXArray? {
        let k = x.dim(-1)
        guard x.size == k, k % 512 == 0, q.rows % 8 == 0,
            q.bits == 4, q.groupSize == 32, q.mode == .affine,
            q.weight.dim(1) * 8 == k, x.dtype == .bfloat16,
            q.scales.dtype == x.dtype, let biases = q.biases,
            biases.dtype == x.dtype
        else { return nil }
        let prepared = prepareKernel(
            [x], template: [("T", x.dtype), ("K", k)],
            grid: (k / 16, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[k], [k / 16]], outputDTypes: [.float32, .float32])
        var shape = x.shape
        shape[shape.count - 1] = q.rows
        return projectKernel(
            [q.weight, q.scales, biases, prepared[0], prepared[1]],
            template: [("T", x.dtype), ("K", k)],
            grid: (32, q.rows / 4, 1), threadGroup: (32, 2, 1),
            outputShapes: [shape], outputDTypes: [x.dtype])[0]
    }
}
