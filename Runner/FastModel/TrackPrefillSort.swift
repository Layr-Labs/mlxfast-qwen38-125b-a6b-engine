// TrackPrefillSort.swift -- the routed-assignment permutation without MLX's
// multi-block merge sort.
//
// WHAT THIS IS. `TrackPrefillIndirect.apply` needs, for a wide window, the
// permutation that groups the `S * top_k` routed assignments by expert id:
//
//     order     = argSort(flatIDs)              // 7 launches (MLX `mbsort`)
//     inverse   = argSort(order)                // 7 more
//     sortedIDs = flatIDs[order]                // a gather
//     tokenRows = order / top_k                 // a divide
//
// At the prefill shape `flatIDs` is 10 240 `uint32` values drawn from 512
// expert ids. MLX sizes its sort from the element count alone: `tn = 4`,
// `bn = 512`, so `n_per_block = 2048`, `n_blocks = 5`, and the merge ladder
// runs `mb_block_sort` plus three (`mb_block_partition`, `mb_block_merge`)
// pairs -- **seven launches per argSort, fourteen per layer, 672 per
// 1024-token chunk**, each one latency-bound on a 40 KB array.
//
// A counting sort over the 512 buckets produces the same permutation in two
// launches, and produces `sortedIDs`, `tokenRows` and `inverse` directly, so
// the gather and the divide go away with it.
//
// EXACTNESS. MLX's argsort is **stable**, and a stable counting sort is the
// unique stable permutation, so the two agree element for element:
//
//   * `ThreadSort::sort` is an odd-even transposition sort whose swap
//     predicate is `op(vals[j+1], vals[j])` -- a STRICT less-than (`a < b`
//     after the NaN screen). Equal keys never swap, so the per-thread sort is
//     stable.
//   * `BlockMergeSort::merge_step` takes from B only when
//     `(b_idx < B_sz) && (a_idx >= A_sz || op(b, a))`, again strict, so on a
//     tie the A element -- the one from the lower-indexed half -- goes first.
//     `merge_partition`'s binary search has the matching bias (`if (op(b, a))
//     A_ed = md; else A_st = md + 1`).
//   * `KernelMultiBlockMergeSort::block_sort` seeds `tgp_idxs[i] = idx` with
//     the GLOBAL index and the multi-block merge ladder always merges a lower
//     index range as A, so the tie order is index order at every level.
//
// Therefore `argSort(flatIDs)` lists, within each expert id, the slot indices
// in increasing order -- which is exactly what
// `base + (# earlier slots with the same id)` computes below. The second
// `argSort(order)` needs no stability argument at all: `order` is a
// permutation of `0 ..< R`, its values are distinct, so its sort is unique
// and `argSort(order)` IS the inverse permutation, which the scatter writes
// directly. `sortedIDs` is tie-order-independent (the sorted multiset), and
// `tokenRows` is `order / top_k` element-wise.
//
// No arithmetic is involved anywhere: every value produced is an index, and
// the kernels below are bit-compared against the four-op MLX chain on the
// real routing indices of every layer.

import Foundation
import MLX
import MLXLMCommon

enum TrackPrefillSort {
    /// `TRACK_PREFILL_COUNTING_SORT=0` restores the `argSort` chain.
    static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_COUNTING_SORT"] != "0"

    /// Threads per block of assignments. 256 keeps the per-block value tile in
    /// threadgroup memory and the predecessor scan at 256 comparisons.
    static let blockSize = 256

    /// Per-block occupancy counts, `[nBlocks, E]`.
    private static let countKernel = MLXFast.metalKernel(
        name: "track_route_block_counts",
        inputNames: ["ids"], outputNames: ["counts"],
        source: countSource, header: "", ensureRowContiguous: true)

    /// Prefixes the count table once and emits the exact tile table consumed
    /// by the indirect kernels. The next scatter launch only reads block bases.
    private static let metadataKernel = MLXFast.metalKernel(
        name: "track_route_metadata",
        inputNames: ["counts"], outputNames: ["block_bases", "tiles"],
        source: metadataSource, header: "", ensureRowContiguous: true)

    /// The stable destination of every assignment, plus the three arrays the
    /// consumers actually read.
    private static let scatterKernel = MLXFast.metalKernel(
        name: "track_route_counting_scatter",
        inputNames: ["ids", "block_bases"],
        outputNames: ["sorted_ids", "token_rows", "inverse"],
        source: scatterSource, header: "", ensureRowContiguous: true)

    /// `(sortedIDs, tokenRows, inverse, tiles)` for `flatIDs` over `E` expert
    /// ids, or nil when the shape is outside the supported window.
    static func apply(flatIDs: MLXArray, experts E: Int, topK: Int)
        -> (sortedIDs: MLXArray, tokenRows: MLXArray, inverse: MLXArray, tiles: MLXArray)?
    {
        let R = flatIDs.size
        guard enabled, flatIDs.ndim == 1, flatIDs.dtype == .uint32, topK > 0,
            E == 512, R % topK == 0,
            R >= blockSize, R % blockSize == 0
        else { return nil }
        let nBlocks = R / blockSize
        let counts = countKernel(
            [flatIDs],
            template: [("E", E), ("BLK", blockSize), ("R", R)],
            grid: (R, 1, 1), threadGroup: (blockSize, 1, 1),
            outputShapes: [[nBlocks * E]], outputDTypes: [.uint32])[0]
        let maxT = (R + 32 - 1) / 32 + E
        let metadata = metadataKernel(
            [counts],
            template: [("E", E), ("BLK", blockSize), ("R", R), ("NB", nBlocks),
                       ("BM", 32), ("MAXT", maxT), ("TG", 1024)],
            grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[nBlocks * E], [2 * maxT]],
            outputDTypes: [.uint32, .uint32])
        let outs = scatterKernel(
            [flatIDs, metadata[0]],
            template: [("E", E), ("BLK", blockSize), ("R", R), ("NB", nBlocks), ("TOPK", topK)],
            grid: (R, 1, 1), threadGroup: (blockSize, 1, 1),
            outputShapes: [[R], [R], [R]],
            outputDTypes: [.uint32, .uint32, .uint32])
        return (outs[0], outs[1], outs[2], metadata[1])
    }

    // MARK: - kernels

    /// One threadgroup per assignment block. The 512-bucket path intersects
    /// ballot bit planes; other geometries scan the original value tile.
    /// Each bucket has one counter owner and neither path uses atomics.
    static let countSource = #"""
        if constexpr (E == 512 && BLK == 256) {
            threadgroup uint planes[8][10];
            const uint blk = threadgroup_position_in_grid.x;
            const uint t = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint value = ids[blk * BLK + t];
            const uint valid = (uint)((simd_vote::vote_t)simd_ballot(value < (uint)E));
            if (lane == 0) { planes[sg][9] = valid; }
            for (uint bit = 0; bit < 9; ++bit) {
                const uint mask = (uint)((simd_vote::vote_t)simd_ballot((value & (1u << bit)) != 0));
                if (lane == 0) { planes[sg][bit] = mask; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint b = t; b < (uint)E; b += BLK) {
                uint count = 0;
                for (uint word = 0; word < 8; ++word) {
                    uint matches = planes[word][9];
                    #pragma clang loop unroll(full)
                    for (uint bit = 0; bit < 9; ++bit) {
                        const uint mask = planes[word][bit];
                        matches &= (b & (1u << bit)) ? mask : ~mask;
                    }
                    count += popcount(matches);
                }
                counts[blk * (uint)E + b] = count;
            }
        } else {
        threadgroup uint vals[BLK];
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint gi = blk * BLK + t;
        vals[t] = (gi < (uint)R) ? ids[gi] : (uint)E;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = t; b < (uint)E; b += BLK) {
            uint c = 0;
            for (uint j = 0; j < BLK; ++j) { c += (vals[j] == b) ? 1u : 0u; }
            counts[blk * (uint)E + b] = c;
        }
        }
        """#

    /// One 1024-threadgroup launch. Thread zero builds expert bases and the
    /// exact 32-row tile table; all threads then materialize per-block bases.
    static let metadataSource = #"""
        threadgroup uint expert_total[E];
        threadgroup uint expert_base[E];
        const uint t = thread_position_in_threadgroup.x;
        if (t == 0) {
            for (uint i = 0; i < (uint)(2 * MAXT); ++i) { tiles[i] = 0; }
            for (uint b = 0; b < (uint)E; ++b) {
                uint total = 0;
                for (uint n = 0; n < (uint)NB; ++n) {
                    total += counts[n * (uint)E + b];
                }
                expert_total[b] = total;
            }
            uint destination = 0;
            uint tile_slot = 0;
            for (uint b = 0; b < (uint)E; ++b) {
                const uint total = expert_total[b];
                expert_base[b] = destination;
                const uint tile_count = (total + (uint)BM - 1) / (uint)BM;
                for (uint j = 0; j < tile_count; ++j) {
                    const uint begin = destination + j * (uint)BM;
                    const uint end = destination + min(total, (j + 1) * (uint)BM);
                    tiles[2 * tile_slot] = begin;
                    tiles[2 * tile_slot + 1] = end;
                    ++tile_slot;
                }
                destination += total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = t; b < (uint)E; b += (uint)TG) {
            uint base = expert_base[b];
            for (uint n = 0; n < (uint)NB; ++n) {
                block_bases[n * (uint)E + b] = base;
                base += counts[n * (uint)E + b];
            }
        }
        """#

    /// One threadgroup per block. The metadata launch already materialized the
    /// destination of each block/expert pair; this launch only computes the
    /// unchanged within-block stable rank and places each assignment.
    ///
    /// `dest = blockBase[block, v] + (# assignments with id v earlier in this
    /// block)`
    ///
    /// is the stable rank of the assignment among its equals, i.e. exactly the
    /// position `argSort` gives it.
    static let scatterSource = #"""
        threadgroup uint vals[BLK];
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint gi = blk * BLK + t;
        vals[t] = (gi < (uint)R) ? ids[gi] : (uint)E;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gi >= (uint)R) { return; }
        const uint v = vals[t];
        uint rank = 0;
        for (uint j = 0; j < t; ++j) { rank += (vals[j] == v) ? 1u : 0u; }
        const uint dest = block_bases[blk * (uint)E + v] + rank;
        sorted_ids[dest] = v;
        token_rows[dest] = gi / (uint)TOPK;
        inverse[gi] = dest;
        """#
}
