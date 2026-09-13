import MLX

enum TrackExpertPacking {
    private static let kernel = MLXFast.metalKernel(
        name: "track_expert_atomic_packing",
        inputNames: ["ids"], outputNames: ["sorted_ids", "token_rows", "inverse", "tiles"],
        source: source, header: "", ensureRowContiguous: true)

    static func apply(flatIDs: MLXArray, experts: Int, topK: Int)
        -> (sortedIDs: MLXArray, tokenRows: MLXArray, inverse: MLXArray, tiles: MLXArray)?
    {
        let rows = flatIDs.size
        guard TrackPrefillSort.enabled, experts == 512, flatIDs.ndim == 1,
            flatIDs.dtype == .uint32, topK > 0, rows % topK == 0,
            rows >= 2048, rows < 32768, rows % 256 == 0
        else { return nil }
        let maxT = TrackPrefillIndirect.maxTiles(rows: rows, experts: experts)
        let out = kernel(
            [flatIDs],
            template: [("R", rows), ("TOPK", topK), ("MAXT", maxT)],
            grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[rows], [rows], [rows], [2 * maxT]],
            outputDTypes: [.uint32, .uint32, .uint32, .uint32])
        return (out[0], out[1], out[2], out[3])
    }

    static let source = #"""
        threadgroup atomic_uint counts[512];
        threadgroup uint starts[512];
        threadgroup uint group_counts[16];
        threadgroup uint group_tiles[16];
        threadgroup uint used_tiles;
        const uint t = thread_position_in_threadgroup.x;
        const uint sg = t / 32;
        if (t < 512) { atomic_store_explicit(&counts[t], 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = t; r < (uint)R; r += 1024) {
            atomic_fetch_add_explicit(&counts[ids[r]], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint count = 0, tile_count = 0, local_start = 0, local_tile = 0;
        if (t < 512) {
            count = atomic_load_explicit(&counts[t], memory_order_relaxed);
            tile_count = (count + 31) / 32;
            local_start = simd_prefix_exclusive_sum(count);
            local_tile = simd_prefix_exclusive_sum(tile_count);
            if ((t % 32) == 31) {
                group_counts[sg] = local_start + count;
                group_tiles[sg] = local_tile + tile_count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 32) {
            const uint c = t < 16 ? group_counts[t] : 0u;
            const uint n = t < 16 ? group_tiles[t] : 0u;
            const uint cbase = simd_prefix_exclusive_sum(c);
            const uint nbase = simd_prefix_exclusive_sum(n);
            if (t < 16) { group_counts[t] = cbase; group_tiles[t] = nbase; }
            if (t == 16) { used_tiles = nbase; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 512) {
            const uint begin = local_start + group_counts[sg];
            const uint slot = local_tile + group_tiles[sg];
            starts[t] = begin;
            atomic_store_explicit(&counts[t], 0u, memory_order_relaxed);
            for (uint j = 0; j < tile_count; ++j) {
                tiles[2 * (slot + j)] = begin + j * 32;
                tiles[2 * (slot + j) + 1] = begin + min(count, (j + 1) * 32);
            }
        }
        for (uint i = used_tiles + t; i < (uint)MAXT; i += 1024) {
            tiles[2 * i] = 0u;
            tiles[2 * i + 1] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = t; r < (uint)R; r += 1024) {
            const uint expert = ids[r];
            const uint rank = atomic_fetch_add_explicit(&counts[expert], 1u, memory_order_relaxed);
            const uint dest = starts[expert] + rank;
            sorted_ids[dest] = expert;
            token_rows[dest] = r / (uint)TOPK;
            inverse[r] = dest;
        }
        """#
}
