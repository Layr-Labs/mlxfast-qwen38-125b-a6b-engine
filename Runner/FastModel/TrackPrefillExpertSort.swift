import MLX
import MLXLMCommon

enum TrackPrefillExpertSort {
    private static let blockSize = 256

    nonisolated(unsafe) private static let histogramKernel = MLXFast.metalKernel(
        name: "track_prefill_expert_histogram",
        inputNames: ["indices"], outputNames: ["counts", "ranks"],
        source: """
            constexpr uint WARPS = BLOCK / 32;
            const uint t = thread_position_in_threadgroup.x;
            const uint block = threadgroup_position_in_grid.x;
            const uint i = block * BLOCK + t;
            const uint warp = t / 32;
            const uint lane = t % 32;
            threadgroup atomic_uint masks[512 * WARPS];
            for (uint j = t; j < 512 * WARPS; j += BLOCK) {
                atomic_store_explicit(&masks[j], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint expert = i < N ? indices[i] : 0u;
            if (i < N) {
                atomic_fetch_or_explicit(
                    &masks[expert * WARPS + warp], 1u << lane, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i < N) {
                uint rank = 0;
                for (uint w = 0; w < warp; ++w) {
                    rank += popcount(atomic_load_explicit(
                        &masks[expert * WARPS + w], memory_order_relaxed));
                }
                const uint peers = atomic_load_explicit(
                    &masks[expert * WARPS + warp], memory_order_relaxed);
                ranks[i] = rank + popcount(peers & ((1u << lane) - 1u));
            }
            for (uint e = t; e < 512; e += BLOCK) {
                uint count = 0;
                for (uint w = 0; w < WARPS; ++w) {
                    count += popcount(atomic_load_explicit(
                        &masks[e * WARPS + w], memory_order_relaxed));
                }
                counts[block * 512 + e] = count;
            }
            """,
        ensureRowContiguous: true)

    nonisolated(unsafe) private static let offsetsKernel = MLXFast.metalKernel(
        name: "track_prefill_expert_offsets",
        inputNames: ["counts"], outputNames: ["offsets"],
        source: """
            const uint e = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint warp = simdgroup_index_in_threadgroup;
            threadgroup uint totals[16];
            uint total = 0;
            for (uint block = 0; block < BLOCKS; ++block) {
                total += counts[block * 512 + e];
            }
            uint base = simd_prefix_exclusive_sum(total);
            if (lane == 31) totals[warp] = base + total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint w = 0; w < warp; ++w) base += totals[w];
            for (uint block = 0; block < BLOCKS; ++block) {
                offsets[block * 512 + e] = base;
                base += counts[block * 512 + e];
            }
            """,
        ensureRowContiguous: true)

    nonisolated(unsafe) private static let scatterKernel = MLXFast.metalKernel(
        name: "track_prefill_expert_permutations",
        inputNames: ["indices", "offsets", "ranks"],
        outputNames: ["order", "inverse"],
        source: """
            const uint i = thread_position_in_grid.x;
            if (i >= N) return;
            const uint position = offsets[(i / BLOCK) * 512 + indices[i]] + ranks[i];
            order[position] = i;
            inverse[i] = position;
            """,
        ensureRowContiguous: true)

    static func sortedInputs(x: MLXArray, indices: MLXArray, expertCount: Int)
        -> (MLXArray, MLXArray, MLXArray)
    {
        guard expertCount == 512, indices.dtype == .uint32,
            indices.size > 2048, indices.size <= 65_536,
            StreamOrDevice.default.stream === Stream.gpu
        else { return gatherSort(x: x, indices: indices) }
        let topK = indices.dim(-1)
        let flat = indices.flattened()
        let count = flat.size
        let blocks = (count + blockSize - 1) / blockSize
        let histogram = histogramKernel(
            [flat], template: [("N", count), ("BLOCK", blockSize)],
            grid: (blocks * blockSize, 1, 1), threadGroup: (blockSize, 1, 1),
            outputShapes: [[blocks, 512], [count]], outputDTypes: [.uint32, .uint32])
        let offsets = offsetsKernel(
            [histogram[0]], template: [("BLOCKS", blocks)],
            grid: (512, 1, 1), threadGroup: (512, 1, 1),
            outputShapes: [[blocks, 512]], outputDTypes: [.uint32])[0]
        let permutations = scatterKernel(
            [flat, offsets, histogram[1]], template: [("N", count), ("BLOCK", blockSize)],
            grid: (count, 1, 1), threadGroup: (blockSize, 1, 1),
            outputShapes: [[count], [count]], outputDTypes: [.uint32, .uint32])
        let order = permutations[0]
        return (
            x.flattened(start: 0, end: -3)[order.floorDivide(topK)],
            flat[order], permutations[1]
        )
    }
}
