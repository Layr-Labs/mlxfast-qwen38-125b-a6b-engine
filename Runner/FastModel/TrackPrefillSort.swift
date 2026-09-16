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

    private static let flashScatterKernel = MLXFast.metalKernel(
        name: "track_flash_counting_scatter_tiles_v3",
        inputNames: ["ids", "counts"],
        outputNames: ["sorted_ids", "token_rows", "inverse", "tiles"],
        source: flashScatterSource, ensureRowContiguous: true)

    /// `(sortedIDs, tokenRows, inverse)` for `flatIDs` over `E` expert ids,
    /// or nil when the shape is outside the supported window.
    static func apply(flatIDs: MLXArray, experts E: Int, topK: Int)
        -> (sortedIDs: MLXArray, tokenRows: MLXArray, inverse: MLXArray, tiles: MLXArray?)?
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
        if E == 512 {
            let maxT = TrackPrefillIndirect.maxTiles(rows: R, experts: E)
            let outs = flashScatterKernel(
                [flatIDs, counts],
                template: [("E", E), ("BLK", blockSize), ("R", R), ("NB", nBlocks),
                           ("TOPK", topK), ("BM", TrackPrefillIndirect.tileRows), ("MAXT", maxT)],
                grid: (R, 1, 1), threadGroup: (blockSize, 1, 1),
                outputShapes: [[R], [R], [R], [2 * maxT]],
                outputDTypes: [.uint32, .uint32, .uint32, .uint32])
            return (outs[0], outs[1], outs[2], outs[3])
        }
        let outs = scatterKernel(
            [flatIDs, counts],
            template: [("E", E), ("BLK", blockSize), ("R", R), ("NB", nBlocks), ("TOPK", topK)],
            grid: (R, 1, 1), threadGroup: (blockSize, 1, 1),
            outputShapes: [[R], [R], [R]],
            outputDTypes: [.uint32, .uint32, .uint32])
        return (outs[0], outs[1], outs[2], nil)
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
    /// Flash has exactly 512 buckets. Two consecutive buckets per thread
    /// make their exclusive scan one SIMD scan plus eight SIMD totals.
    /// Each assignment's stable predecessor count uses nine ballot planes.
    /// Only block zero emits expert-aligned tiles, directly from bucket counts.
    static let flashScatterSource = #"""
        static_assert(E == 512 && BLK == 256, "Flash routing geometry");
        threadgroup uint before[512], bases[512];
        threadgroup uint group_totals[8];
        threadgroup uint planes[8][9];
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint gi = blk * 256 + t;
        const uint v = ids[gi];
        for (uint bit = 0; bit < 9; ++bit) {
            const uint mask = uint((simd_vote::vote_t)simd_ballot((v & (1u << bit)) != 0));
            if (lane == 0) { planes[sg][bit] = mask; }
        }
        uint n[2], prior[2];
        for (uint j = 0; j < 2; ++j) {
            const uint expert = 2 * t + j;
            n[j] = 0; prior[j] = 0;
            for (uint block = 0; block < NB; ++block) {
                const uint c = counts[block * 512 + expert];
                n[j] += c;
                if (block < blk) { prior[j] += c; }
            }
            before[expert] = prior[j];
        }
        const uint pair_total = n[0] + n[1];
        const uint local_base = simd_prefix_exclusive_sum(pair_total);
        if (lane == 31) { group_totals[sg] = local_base + pair_total; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint base = local_base;
        for (uint group = 0; group < sg; ++group) { base += group_totals[group]; }
        bases[2 * t] = base;
        bases[2 * t + 1] = base + n[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint rank = 0;
        for (uint group = 0; group <= sg; ++group) {
            uint matches = 0xffffffffu;
            for (uint bit = 0; bit < 9; ++bit) {
                const uint mask = planes[group][bit];
                matches &= (v & (1u << bit)) ? mask : ~mask;
            }
            if (group == sg) { matches &= (1u << lane) - 1u; }
            rank += popcount(matches);
        }
        const uint dest = bases[v] + before[v] + rank;
        sorted_ids[dest] = v;
        token_rows[dest] = gi / TOPK;
        inverse[gi] = dest;
        if (blk == 0) {
            // Protect group_totals until every lane finished the first scan.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint tiles0 = (n[0] + BM - 1) / BM;
            const uint tiles1 = (n[1] + BM - 1) / BM;
            const uint pair_tiles = tiles0 + tiles1;
            const uint local_slot = simd_prefix_exclusive_sum(pair_tiles);
            if (lane == 31) { group_totals[sg] = local_slot + pair_tiles; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint slot = local_slot, total = 0;
            for (uint group = 0; group < 8; ++group) {
                if (group < sg) { slot += group_totals[group]; }
                total += group_totals[group];
            }
            for (uint j = 0; j < 2; ++j) {
                const uint start = base + (j == 0 ? 0u : n[0]);
                const uint count = j == 0 ? tiles0 : tiles1;
                for (uint tile = 0; tile < count; ++tile) {
                    tiles[2 * (slot + tile)] = start + tile * BM;
                    tiles[2 * (slot + tile) + 1] = start + min(n[j], (tile + 1) * BM);
                }
                slot += count;
            }
            for (uint tile = total + t; tile < MAXT; tile += 256) {
                tiles[2 * tile] = 0; tiles[2 * tile + 1] = 0;
            }
        }
        """#

    static let scatterSource = #"""
        threadgroup uint vals[BLK];
        threadgroup uint tot[E];
        threadgroup uint pre[E];
        threadgroup uint sA[E];
        threadgroup uint sB[E];
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = thread_position_in_threadgroup.x;
        const uint gi = blk * BLK + t;
        vals[t] = (gi < (uint)R) ? ids[gi] : (uint)E;
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
        for (uint j = 0; j < t; ++j) { rank += (vals[j] == v) ? 1u : 0u; }
        const uint dest = (sA[v] - tot[v]) + pre[v] + rank;
        sorted_ids[dest] = v;
        token_rows[dest] = gi / (uint)TOPK;
        inverse[gi] = dest;
        """#
}
