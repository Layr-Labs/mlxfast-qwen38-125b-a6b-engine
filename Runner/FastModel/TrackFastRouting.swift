import MLX

/// Small-window expert selection and normalization over router logits.
enum TrackFastRouting {
    static let source = #"""
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.x;
        uint keys[ITEMS];
        for (uint j = 0; j < ITEMS; ++j) {
            const uint bits = as_type<uint>(logits[row * E + j * 32 + lane]);
            const uint magnitude = bits & 0x7fffffffu;
            // MLX's stable ascending sort of -logits puts NaNs last and
            // keeps input order on equal values, including the two zeros.
            keys[j] = magnitude > 0x7f800000u ? 0u
                : magnitude == 0u ? 0x80000000u
                : (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        }
        uint remaining = 0xffffffffu >> (32 - ITEMS);
        float selected[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (uint rank = 0; rank < K; ++rank) {
            uint bestKey = 0u;
            uint bestId = 0xffffffffu;
            for (uint j = 0; j < ITEMS; ++j) {
                const uint id = j * 32 + lane;
                const uint key = keys[j];
                if ((remaining & (1u << j)) &&
                    (key > bestKey || (key == bestKey && id < bestId))) {
                    bestKey = key;
                    bestId = id;
                }
            }
            const uint winningKey = simd_max(bestKey);
            const uint winner = simd_min(bestKey == winningKey ? bestId : 0xffffffffu);
            if (lane == (winner & 31u)) {
                remaining &= ~(1u << (winner / 32u));
            }
            if (lane == 0) indices[row * K + rank] = winner;
            if (lane == rank / 4u) {
                selected[rank % 4u] = logits[row * E + winner];
            }
        }

        // Reproduce softmax_single_row<float, float, 4>: four consecutive
        // selected logits per lane, the same two SIMD reductions, fast exp,
        // reciprocal, and final multiplication.
        float maxval = -3.402823466e+38f;
        for (uint j = 0; j < 4; ++j) {
            maxval = maxval < selected[j] ? selected[j] : maxval;
        }
        maxval = simd_max(maxval);
        maxval = simd_max(lane == 0 ? maxval : -INFINITY);
        float normalizer = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            selected[j] = fast::exp(selected[j] - maxval);
            normalizer += selected[j];
        }
        normalizer = simd_sum(normalizer);
        normalizer = simd_sum(lane == 0 ? normalizer : 0.0f);
        normalizer = 1.0f / normalizer;
        for (uint j = 0; j < 4; ++j) {
            const uint rank = lane * 4u + j;
            if (rank < K) weights[row * K + rank] = selected[j] * normalizer;
        }
        """#

    static let kernel = MLXFast.metalKernel(
        name: "track_moe_route", inputNames: ["logits"],
        outputNames: ["indices", "weights"], source: source,
        ensureRowContiguous: true)

    static func route(_ logits: MLXArray, topK: Int) -> (indices: MLXArray, weights: MLXArray)? {
        guard logits.dtype == .float32, logits.ndim == 3,
            logits.dim(0) == 1, (1 ... 8).contains(logits.dim(1)),
            logits.dim(2) > 0, logits.dim(2) <= 1024, logits.dim(2) % 32 == 0,
            topK > 0, topK <= 32, topK <= logits.dim(2)
        else { return nil }
        let experts = logits.dim(2), rows = logits.dim(1)
        let output = kernel(
            [logits], template: [("E", experts), ("ITEMS", experts / 32), ("K", topK)],
            grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[1, rows, topK], [1, rows, topK]],
            outputDTypes: [.uint32, .float32])
        return (output[0], output[1])
    }
}
