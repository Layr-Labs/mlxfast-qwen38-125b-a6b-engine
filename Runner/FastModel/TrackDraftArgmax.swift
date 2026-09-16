import MLX

/// Proposal-side substitute for a coarse LM-head screen: evaluate every row
/// in the existing native-head shortlist with the same affine QMV, but keep
/// only exact BF16 block winners. No approximation or target-logit pruning.
enum TrackDraftArgmax {
    private static let blocksKernel = MLXFast.metalKernel(
        name: "track_draft_qmv_block_argmax_v3",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["values", "indices"], source: """
            const uint block = threadgroup_position_in_grid.x;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint first = block * 32 + sg * 4;
            float best = -INFINITY;
            uint index = 0xffffffffu;
            if (first < N) {
                float result[4];
                qmv_fast_reg<T, GS, 4, 4>(w, scales, biases, x, K, first, lane, result);
                for (uint j = 0; j < 4; ++j) {
                    const float value = float(T(result[j]));
                    if (value > best || (value == best && first + j < index)) {
                        best = value; index = first + j;
                    }
                }
            }
            threadgroup float local_values[8];
            threadgroup uint local_indices[8];
            if (lane == 0) { local_values[sg] = best; local_indices[sg] = index; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                best = lane < 8 ? local_values[lane] : -INFINITY;
                index = lane < 8 ? local_indices[lane] : 0xffffffffu;
                const float winner = simd_max(best);
                const uint winner_index = simd_min(best == winner ? index : 0xffffffffu);
                if (lane == 0) { values[block] = winner; indices[block] = winner_index; }
            }
            """, header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
            + TrackFastMoEKernels.regHelpers,
        ensureRowContiguous: true)

    private static let finishKernel = MLXFast.metalKernel(
        name: "track_draft_exact_argmax_finish_v3",
        inputNames: ["values", "indices"], outputNames: ["winner"], source: """
            const uint t = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            float best = -INFINITY;
            uint index = 0xffffffffu;
            for (uint i = t; i < BLOCKS; i += 256) {
                const float value = values[i]; const uint candidate = indices[i];
                if (value > best || (value == best && candidate < index)) {
                    best = value; index = candidate;
                }
            }
            const float maximum = simd_max(best);
            index = simd_min(best == maximum ? index : 0xffffffffu);
            threadgroup float local_values[8];
            threadgroup uint local_indices[8];
            if (lane == 0) { local_values[sg] = maximum; local_indices[sg] = index; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                best = lane < 8 ? local_values[lane] : -INFINITY;
                index = lane < 8 ? local_indices[lane] : 0xffffffffu;
                const float value = simd_max(best);
                const uint id = simd_min(best == value ? index : 0xffffffffu);
                if (lane == 0) { winner[0] = id == 0xffffffffu ? 0u : id; }
            }
            """, ensureRowContiguous: true)

    static func apply(
        x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        groupSize: Int, bits: Int
    ) -> MLXArray? {
        guard StreamOrDevice.default.stream === Stream.gpu,
            x.shape == [1, 2560], x.dtype == .bfloat16, bits == 4,
            groupSize == 32 || groupSize == 64,
            weight.ndim == 2, weight.dim(1) == 320, weight.dim(0) >= 32,
            weight.dim(0) % 8 == 0, weight.dtype == .uint32,
            scales.shape == [weight.dim(0), 2560 / groupSize], biases.shape == scales.shape,
            scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        let blocks = (weight.dim(0) + 31) / 32
        let partials = blocksKernel(
            [x, weight, scales, biases],
            template: [("T", x.dtype), ("GS", groupSize), ("K", 2560), ("N", weight.dim(0))],
            grid: (blocks * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[blocks], [blocks]], outputDTypes: [.float32, .uint32])
        return finishKernel(
            partials, template: [("BLOCKS", blocks)],
            grid: (256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1]], outputDTypes: [.uint32])[0]
    }
}
