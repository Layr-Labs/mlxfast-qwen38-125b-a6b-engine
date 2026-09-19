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

    /// The stable destination of every assignment, plus the three arrays the
    /// consumers actually read.
    private static let scatterKernel = MLXFast.metalKernel(
        name: "track_route_counting_scatter",
        inputNames: ["ids", "counts"],
        outputNames: ["sorted_ids", "token_rows", "inverse"],
        source: scatterSource, header: "", ensureRowContiguous: true)

    /// `(sortedIDs, tokenRows, inverse)` for `flatIDs` over `E` expert ids,
    /// or nil when the shape is outside the supported window.
    static func apply(flatIDs: MLXArray, experts E: Int, topK: Int)
        -> (sortedIDs: MLXArray, tokenRows: MLXArray, inverse: MLXArray)?
    {
        let R = flatIDs.size
        guard enabled, flatIDs.ndim == 1, flatIDs.dtype == .uint32, topK > 0,
            E > 0, E % blockSize == 0, E <= 4096, R % topK == 0,
            R >= blockSize, R % blockSize == 0
        else { return nil }
        let nBlocks = R / blockSize
        let counts = countKernel(
            [flatIDs],
            template: [("E", E), ("BLK", blockSize), ("R", R)],
            grid: (R, 1, 1), threadGroup: (blockSize, 1, 1),
            outputShapes: [[nBlocks * E]], outputDTypes: [.uint32])[0]
        let outs = scatterKernel(
            [flatIDs, counts],
            template: [("E", E), ("BLK", blockSize), ("R", R), ("NB", nBlocks), ("TOPK", topK)],
            grid: (R, 1, 1), threadGroup: (blockSize, 1, 1),
            outputShapes: [[R], [R], [R]],
            outputDTypes: [.uint32, .uint32, .uint32])
        return (outs[0], outs[1], outs[2])
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

    /// One threadgroup per block again. Each threadgroup re-derives the whole
    /// bucket base table (`E` exclusive-scanned totals) and its own block's
    /// running offset from the `[NB, E]` count table -- `NB * E` reads, which
    /// is cheaper than a third launch -- then places its own elements.
    ///
    /// `dest = bucketBase[v] + (# assignments with id v in earlier blocks)
    ///                       + (# assignments with id v earlier in this block)`
    ///
    /// is the stable rank of the assignment among its equals, i.e. exactly the
    /// position `argSort` gives it.
    static let scatterSource = #"""
        threadgroup uint vals[BLK];
        threadgroup uint tot[E];
        threadgroup uint pre[E];
        threadgroup uint sA[E];
        threadgroup uint sB[E];
        threadgroup uint planes[8][10];
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint gi = blk * BLK + t;
        const uint value = (gi < (uint)R) ? ids[gi] : (uint)E;
        vals[t] = value;
        // MLXFAST-SCATTERPLANES: the same ballot bit planes `countSource`
        // builds, reused below to rank this element among its equals in
        // O(simdgroups) instead of the O(BLK) serial scan. Plane 9 is the
        // validity bit (value < E) so sentinel rows can never alias value 0.
        if constexpr (E == 512 && BLK == 256) {
            const uint p_lane = thread_index_in_simdgroup;
            const uint p_sg = simdgroup_index_in_threadgroup;
            if (p_lane == 0) { planes[p_sg][9] = (uint)((simd_vote::vote_t)simd_ballot(value < (uint)E)); }
            #pragma clang loop unroll(full)
            for (uint bit = 0; bit < 9; ++bit) {
                const uint mask = (uint)((simd_vote::vote_t)simd_ballot((value & (1u << bit)) != 0));
                if (p_lane == 0) { planes[p_sg][bit] = mask; }
            }
        }
        for (uint b = t; b < (uint)E; b += BLK) {
            uint s = 0, before = 0;
            for (uint n = 0; n < (uint)NB; ++n) {
                const uint c = counts[n * (uint)E + b];
                if (n < blk) { before += c; }
                s += c;
            }
            tot[b] = s;
            pre[b] = before;
            sA[b] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1; off < (uint)E; off <<= 1) {
            for (uint b = t; b < (uint)E; b += BLK) {
                sB[b] = sA[b] + ((b >= off) ? sA[b - off] : 0u);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint b = t; b < (uint)E; b += BLK) { sA[b] = sB[b]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (gi >= (uint)R) { return; }
        const uint v = vals[t];
        uint rank = 0;
        if constexpr (E == 512 && BLK == 256) {
            // Rank among equals = popcount of matching lanes before `t`.
            // The bit-9 validity plane keeps sentinel rows out of value 0's
            // set, so this counts exactly the j < t with vals[j] == v.
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            for (uint g = 0; g < sg; ++g) {
                uint matches = planes[g][9];
                #pragma clang loop unroll(full)
                for (uint bit = 0; bit < 9; ++bit) {
                    const uint mask = planes[g][bit];
                    matches &= (v & (1u << bit)) ? mask : ~mask;
                }
                rank += popcount(matches);
            }
            uint matches = planes[sg][9] & ((1u << lane) - 1u);
            #pragma clang loop unroll(full)
            for (uint bit = 0; bit < 9; ++bit) {
                const uint mask = planes[sg][bit];
                matches &= (v & (1u << bit)) ? mask : ~mask;
            }
            rank += popcount(matches);
        } else {
            for (uint j = 0; j < t; ++j) { rank += (vals[j] == v) ? 1u : 0u; }
        }
        const uint dest = (sA[v] - tot[v]) + pre[v] + rank;
        sorted_ids[dest] = v;
        token_rows[dest] = gi / (uint)TOPK;
        inverse[gi] = dest;
        """#
}
