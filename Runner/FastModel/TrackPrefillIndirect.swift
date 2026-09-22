import Foundation
import MLX
import MLXLMCommon

enum TrackPrefillIndirect {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_INDIRECT_ACTIVATIONS"] != "0"

    private static let supportsNAX: Bool = {
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return false
        }
        let arch = GPU.deviceInfo().architecture
        guard let generation = Int(arch.dropLast().suffix(2)), let family = arch.last else {
            return false
        }
        return generation >= (family == "p" ? 18 : 17)
    }()

    /// Gate, up and `silu(gate) * up` in one launch over the tile table (two
    /// weight streams, one activation staging, the product in the epilogue);
    /// `kernel` below serves the down projection.
    private static let gateUpKernel = MLXFast.metalKernel(
        name: "track_prefill_indirect_gate_up",
        inputNames: ["x", "w0", "scales0", "biases0", "w1", "scales1", "biases1", "indices", "token_rows", "tiles"],
        outputNames: ["y"], source: sourceGU.replacingOccurrences(of: "y0, y1, N, K", with: "y, y, N, K"),
        header: metalHeader, ensureRowContiguous: true)

    /// Down projection: 128x32 weight blocks (20 column tiles over N = 2560,
    /// 20 K steps over K = 640); the 64x64 block `source` remains the reference.
    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_indirect_down",
        inputNames: ["x", "w", "scales", "biases", "indices", "token_rows", "tiles"],
        outputNames: ["y"], source: sourceDown, header: metalHeader,
        ensureRowContiguous: true)

    /// `[2 * maxTiles]` of `[begin, end)` row ranges, one 32-row tile per slot,
    /// each aligned to the start of its expert run; unused slots are `[0, 0)`.
    private static let tileKernel = MLXFast.metalKernel(
        name: "track_prefill_tile_table",
        inputNames: ["sorted_ids"], outputNames: ["tiles"],
        source: tileSource, header: "", ensureRowContiguous: true)

    /// The two gate/up outputs own disjoint rows. Only `activated` materializes
    /// a full array, and only the fallback caller asks for it.
    struct SplitProjection {
        let shortActivated: MLXArray
        let longActivated: MLXArray
        let sortedIDs: MLXArray
        let inverse: MLXArray
        let tiles: MLXArray
        let tileKind: MLXArray

        var activated: MLXArray {
            let rows = sortedIDs.size
            return TrackPrefillIndirect.mergeKernel(
                [shortActivated, longActivated, tiles, tileKind],
                template: [("T", shortActivated.dtype), ("N", 640)],
                grid: (32 * 640, tileKind.size, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, 1, 640]], outputDTypes: [.bfloat16])[0]
        }
    }

    private static let splitPlanKernel = MLXFast.metalKernel(
        name: "track_prefill_split_tile_plan",
        inputNames: ["sorted_ids"], outputNames: ["tiles32", "tiles64", "tile_kind", "long_count"],
        source: splitPlanSource, header: "", ensureRowContiguous: true)

    private static let shortGateUpKernel = MLXFast.metalKernel(
        name: "track_prefill_short_gate_up",
        inputNames: ["x", "w0", "scales0", "biases0", "w1", "scales1", "biases1", "indices", "token_rows", "tiles", "tile_kind"],
        outputNames: ["y"],
        source: shortPrefix + "\n" + sourceGU.replacingOccurrences(of: "y0, y1, N, K", with: "y, y, N, K"),
        header: metalHeader, ensureRowContiguous: true)

    private static let longGateUpKernel = MLXFast.metalKernel(
        name: "track_prefill_long_gate_up",
        inputNames: ["x", "w0", "scales0", "biases0", "w1", "scales1", "biases1", "indices", "token_rows", "tiles", "long_count"],
        outputNames: ["y"], source: longSourceGU,
        header: metalHeader, ensureRowContiguous: true)

    private static let splitDownKernel = MLXFast.metalKernel(
        name: "track_prefill_split_down",
        inputNames: ["x_short", "x_long", "w", "scales", "biases", "indices", "token_rows", "tiles", "tile_kind"],
        outputNames: ["y"], source: splitDownPrefix + "\n" + sourceDown,
        header: metalHeader, ensureRowContiguous: true)

    private static let mergeKernel = MLXFast.metalKernel(
        name: "track_prefill_split_merge",
        inputNames: ["x_short", "x_long", "tiles", "tile_kind"], outputNames: ["y"],
        source: mergeSource, header: "", ensureRowContiguous: true)

    static let tileRows = 32
    static let tileThreads = 1024

    /// Every expert run of `r` rows takes `ceil(r / 32)` tiles, so the count is
    /// bounded by `rows / 32 + experts` for sorted ids over `experts` values.
    static func maxTiles(rows: Int, experts: Int) -> Int {
        (rows + tileRows - 1) / tileRows + experts
    }

    static func tileTable(sortedIDs: MLXArray, rows: Int, experts: Int) -> MLXArray {
        let maxT = maxTiles(rows: rows, experts: experts)
        return tileKernel(
            [sortedIDs],
            template: [("R", rows), ("E", experts), ("BM", tileRows), ("MAXT", maxT), ("TG", tileThreads)],
            grid: (tileThreads, 1, 1), threadGroup: (tileThreads, 1, 1),
            outputShapes: [[2 * maxT]], outputDTypes: [.uint32])[0]
    }

    static func apply(_ m: TrackMoE, x: MLXArray, indices: MLXArray)
        -> SplitProjection?
    {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            m.expertBits == 4, m.expertGroupSize == 32,
            x.ndim == 3, x.dim(0) == 1, x.dim(2) == 2560, x.dtype == .bfloat16,
            indices.ndim == 3, indices.dim(0) == 1, indices.dim(1) == x.dim(1),
            indices.dim(2) > 0, indices.dtype == .uint32,
            indices.size >= 2048, indices.size < 512 * 64
        else { return nil }
        let g = m.expertGate
        let u = m.expertUp
        guard g.w.shape == [512, 640, 320], u.w.shape == g.w.shape,
            g.s.shape == [512, 640, 80], g.b.shape == g.s.shape,
            u.s.shape == g.s.shape, u.b.shape == g.s.shape,
            g.w.dtype == .uint32, u.w.dtype == .uint32,
            g.s.dtype == .bfloat16, g.b.dtype == .bfloat16,
            u.s.dtype == .bfloat16, u.b.dtype == .bfloat16
        else { return nil }

        let flatIDs = indices.flattened()
        let sortedIDs: MLXArray
        let inverse: MLXArray
        let tokenRows: MLXArray
        if let c = TrackPrefillSort.apply(
            flatIDs: flatIDs, experts: g.w.dim(0), topK: indices.dim(2))
        {
            // The identical permutation in two launches (see TrackPrefillSort).
            (sortedIDs, tokenRows, inverse) = (c.sortedIDs, c.tokenRows, c.inverse)
        } else {
            let order = argSort(flatIDs)
            inverse = argSort(order)
            sortedIDs = flatIDs[order]
            tokenRows = order.floorDivide(indices.dim(2))
        }
        let rows = indices.size
        let experts = g.w.dim(0)
        let maxT = maxTiles(rows: rows, experts: experts)
        let maxLong = (rows + 63) / 64 + experts
        let longGroups = min(maxLong, 64)
        let plan = splitPlanKernel(
            [sortedIDs],
            template: [("R", rows), ("E", experts), ("MAX32", maxT), ("MAX64", maxLong), ("TG", tileThreads)],
            grid: (tileThreads, 1, 1), threadGroup: (tileThreads, 1, 1),
            outputShapes: [[2 * maxT], [2 * maxLong], [maxT], [1]],
            outputDTypes: [.uint32, .uint32, .uint32, .uint32])
        let shortActivated = shortGateUpKernel(
            [x, g.w, g.s, g.b, u.w, u.s, u.b, sortedIDs, tokenRows, plan[0], plan[2]],
            template: [("T", x.dtype), ("N", 640), ("K", 2560), ("SILU", true)],
            grid: (10 * 32, maxT * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 640]], outputDTypes: [.bfloat16])[0]
        let longActivated = longGateUpKernel(
            [x, g.w, g.s, g.b, u.w, u.s, u.b, sortedIDs, tokenRows, plan[1], plan[3]],
            template: [("T", x.dtype), ("N", 640), ("K", 2560), ("SILU", true), ("LONG_GROUPS", longGroups)],
            grid: (10 * 32, longGroups * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 640]], outputDTypes: [.bfloat16])[0]
        return SplitProjection(
            shortActivated: shortActivated, longActivated: longActivated,
            sortedIDs: sortedIDs, inverse: inverse, tiles: plan[0], tileKind: plan[2])
    }

    /// The down projection over the same tile table. `activated` already holds
    /// one row per sorted assignment, so the row gather is the identity.
    static func down(_ m: TrackMoE, activated: MLXArray, sortedIDs: MLXArray, tiles: MLXArray)
        -> MLXArray?
    {
        let d = m.expertDown
        let rows = sortedIDs.size
        guard activated.ndim == 3, activated.dim(0) == rows, activated.dim(1) == 1,
            activated.dim(2) == 640, activated.dtype == .bfloat16,
            d.w.shape == [512, 2560, 80], d.s.shape == [512, 2560, 20], d.b.shape == d.s.shape,
            d.w.dtype == .uint32, d.s.dtype == .bfloat16, d.b.dtype == .bfloat16
        else { return nil }
        let maxT = maxTiles(rows: rows, experts: d.w.dim(0))
        return kernel(
            [activated, d.w, d.s, d.b, sortedIDs, sortedIDs, tiles],
            template: [("T", activated.dtype), ("N", 2560), ("K", 640)],
            grid: ((2560 / downBlockN) * 32, maxT * 2, 2),
            threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 2560]], outputDTypes: [.bfloat16])[0]
    }

    /// Select the one initialized activation buffer uniformly for each original
    /// 32-row down tile. No merge/copy is evaluated on this fast path.
    static func down(_ m: TrackMoE, split: SplitProjection) -> MLXArray? {
        let d = m.expertDown
        let rows = split.sortedIDs.size
        guard d.w.shape == [512, 2560, 80], d.s.shape == [512, 2560, 20], d.b.shape == d.s.shape,
            d.w.dtype == .uint32, d.s.dtype == .bfloat16, d.b.dtype == .bfloat16
        else { return nil }
        let maxT = maxTiles(rows: rows, experts: d.w.dim(0))
        return splitDownKernel(
            [split.shortActivated, split.longActivated, d.w, d.s, d.b,
             split.sortedIDs, split.sortedIDs, split.tiles, split.tileKind],
            template: [("T", split.shortActivated.dtype), ("N", 2560), ("K", 640)],
            grid: ((2560 / downBlockN) * 32, maxT * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 2560]], outputDTypes: [.bfloat16])[0]
    }

    static let shortPrefix = #"""
        const uint slot = threadgroup_position_in_grid.y;
        if (tiles[2 * slot] == tiles[2 * slot + 1] || tile_kind[slot] != 0u) {
            return;
        }
        """#

    static let longSourceGU = #"""
        alignas(16) threadgroup T Ws0[64 * 72];
        alignas(16) threadgroup T Ws1[64 * 72];
        alignas(16) threadgroup T As[64 * 72];
        // The plan contains only tiles with more than 32 live rows. A bounded
        // grid walks that compact list without a host readback or empty-slot grid.
        uint3 tile = threadgroup_position_in_grid;
        const uint count = long_count[0];
        for (uint slot = tile.y; slot < count; slot += (uint)LONG_GROUPS) {
            tile.y = slot;
            track_prefill_indirect_gu_pair<T, 32, 4, 64, 64, 64, 2, 2, true, SILU, N>(
                x, w0, scales0, biases0, w1, scales1, biases1, indices, token_rows, tiles,
                y, y, N, K, Ws0, Ws1, As, tile,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        }
        """#

    static let splitDownPrefix = #"""
        const uint slot = threadgroup_position_in_grid.y;
        if (tiles[2 * slot] == tiles[2 * slot + 1]) { return; }
        const device T* x = tile_kind[slot] != 0u ? x_long : x_short;
        """#

    static let mergeSource = #"""
        const uint slot = thread_position_in_grid.y;
        const uint begin = tiles[2 * slot];
        const uint end = tiles[2 * slot + 1];
        const uint row = begin + thread_position_in_grid.x / (uint)N;
        if (row >= end) { return; }
        const uint col = thread_position_in_grid.x % (uint)N;
        const device T* x = tile_kind[slot] != 0u ? x_long : x_short;
        const size_t offset = size_t(row) * N + col;
        y[offset] = x[offset];
        """#

    static let sourceGU = #"""
        alignas(16) threadgroup T Ws0[64 * 72];
        alignas(16) threadgroup T Ws1[64 * 72];
        alignas(16) threadgroup T As[32 * 72];
        track_prefill_indirect_gu<T, 32, 4, 32, 64, 64, 2, 2, true, SILU, N>(
            x, w0, scales0, biases0, w1, scales1, biases1, indices, token_rows, tiles,
            y0, y1, N, K, Ws0, Ws1, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

    static let downBlockN = 128

    static let sourceDown = #"""
        alignas(16) threadgroup T Ws[128 * 40];
        alignas(16) threadgroup T As[32 * 40];
        track_prefill_indirect<T, 32, 4, 32, 128, 32, 2, 2, true, N, true>(
            x, w, scales, biases, indices, token_rows, tiles, y,
            N, K, Ws, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

    static let source = #"""
        alignas(16) threadgroup T Ws[64 * 72];
        alignas(16) threadgroup T As[32 * 72];
        track_prefill_indirect<T, 32, 4, 32, 64, 64, 2, 2, true, N>(
            x, w, scales, biases, indices, token_rows, tiles, y,
            N, K, Ws, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

    /// One threadgroup of TG threads over the R sorted ids. Pass 1 finds the
    /// run starts (id differs from its predecessor) and ranks them with a scan;
    /// pass 2 gives run r `ceil(len / BM)` tiles, ranks those with a second
    /// scan and writes each tile's [begin, end). Runs are at most E (one per
    /// distinct id) so pass 2 is one run per thread. Deterministic: no atomics.
    static let tileSource = #"""
        static_assert(E <= TG, "one run per thread in pass 2");
        threadgroup uint run_begin[E + 1];
        threadgroup uint sg_runs[TG / 32];
        threadgroup uint sg_tiles[TG / 32];
        const uint t = thread_position_in_threadgroup.x;
        const uint sg = t / 32;
        constexpr uint PER = ((uint)R + (uint)TG - 1) / (uint)TG;
        const uint i0 = t * PER;
        const uint i1 = min(i0 + PER, (uint)R);
        uint starts = 0;
        for (uint i = i0; i < i1; ++i) {
            starts += (i == 0 || sorted_ids[i] != sorted_ids[i - 1]) ? 1u : 0u;
        }
        const uint ex = simd_prefix_exclusive_sum(starts);
        if ((t % 32) == 31) { sg_runs[sg] = ex + starts; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = ex;
        uint nruns = 0;
        for (uint g = 0; g < (uint)(TG / 32); ++g) {
            if (g < sg) { r += sg_runs[g]; }
            nruns += sg_runs[g];
        }
        for (uint i = i0; i < i1; ++i) {
            if (i == 0 || sorted_ids[i] != sorted_ids[i - 1]) { run_begin[r++] = i; }
        }
        if (t == 0) { run_begin[nruns] = (uint)R; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint len = 0;
        uint ntile = 0;
        if (t < nruns) {
            len = run_begin[t + 1] - run_begin[t];
            ntile = (len + (uint)BM - 1) / (uint)BM;
        }
        const uint ex2 = simd_prefix_exclusive_sum(ntile);
        if ((t % 32) == 31) { sg_tiles[sg] = ex2 + ntile; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint slot = ex2;
        uint total = 0;
        for (uint g = 0; g < (uint)(TG / 32); ++g) {
            if (g < sg) { slot += sg_tiles[g]; }
            total += sg_tiles[g];
        }
        if (t < nruns) {
            const uint b = run_begin[t];
            for (uint j = 0; j < ntile; ++j) {
                tiles[2 * (slot + j)] = b + j * (uint)BM;
                tiles[2 * (slot + j) + 1] = b + min(len, (j + 1) * (uint)BM);
            }
        }
        for (uint i = total + t; i < (uint)MAXT; i += (uint)TG) {
            tiles[2 * i] = 0u;
            tiles[2 * i + 1] = 0u;
        }
        """#
    /// Build both tables and their shared ownership map in one launch.
    static let splitPlanSource = #"""
        static_assert(E <= TG, "one run per thread in pass 2");
        threadgroup uint run_begin[E + 1];
        threadgroup uint sg_runs[TG / 32];
        threadgroup uint sg_tiles32[TG / 32];
        threadgroup uint sg_tiles64[TG / 32];
        const uint t = thread_position_in_threadgroup.x;
        const uint sg = t / 32;
        constexpr uint PER = ((uint)R + (uint)TG - 1) / (uint)TG;
        const uint i0 = t * PER;
        const uint i1 = min(i0 + PER, (uint)R);
        uint starts = 0;
        for (uint i = i0; i < i1; ++i) {
            starts += (i == 0 || sorted_ids[i] != sorted_ids[i - 1]) ? 1u : 0u;
        }
        const uint ex = simd_prefix_exclusive_sum(starts);
        if ((t % 32) == 31) { sg_runs[sg] = ex + starts; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = ex;
        uint nruns = 0;
        for (uint g = 0; g < (uint)(TG / 32); ++g) {
            if (g < sg) { r += sg_runs[g]; }
            nruns += sg_runs[g];
        }
        for (uint i = i0; i < i1; ++i) {
            if (i == 0 || sorted_ids[i] != sorted_ids[i - 1]) { run_begin[r++] = i; }
        }
        if (t == 0) { run_begin[nruns] = (uint)R; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint len = 0;
        uint count32 = 0;
        uint count64 = 0;
        if (t < nruns) {
            len = run_begin[t + 1] - run_begin[t];
            count32 = (len + 31u) / 32u;
            count64 = (len + 31u) / 64u; // exclude a last tile with <= 32 rows
        }
        const uint ex32 = simd_prefix_exclusive_sum(count32);
        const uint ex64 = simd_prefix_exclusive_sum(count64);
        if ((t % 32) == 31) {
            sg_tiles32[sg] = ex32 + count32;
            sg_tiles64[sg] = ex64 + count64;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint slot32 = ex32, slot64 = ex64;
        uint total32 = 0, total64 = 0;
        for (uint g = 0; g < (uint)(TG / 32); ++g) {
            if (g < sg) {
                slot32 += sg_tiles32[g];
                slot64 += sg_tiles64[g];
            }
            total32 += sg_tiles32[g];
            total64 += sg_tiles64[g];
        }
        if (t == 0) { long_count[0] = total64; }
        if (t < nruns) {
            const uint b = run_begin[t];
            for (uint j = 0; j < count32; ++j) {
                tiles32[2 * (slot32 + j)] = b + j * 32u;
                tiles32[2 * (slot32 + j) + 1] = b + min(len, (j + 1u) * 32u);
                // Both 32-row halves of a long 64-row tile use its output.
                // An unpaired final half belongs to the original short kernel.
                const uint parent = (j / 2u) * 64u;
                tile_kind[slot32 + j] = len - parent > 32u ? 1u : 0u;
            }
            for (uint j = 0; j < count64; ++j) {
                tiles64[2 * (slot64 + j)] = b + j * 64u;
                tiles64[2 * (slot64 + j) + 1] = b + min(len, (j + 1u) * 64u);
            }
        }
        for (uint i = total32 + t; i < (uint)MAX32; i += (uint)TG) {
            tiles32[2 * i] = 0u;
            tiles32[2 * i + 1] = 0u;
            tile_kind[i] = 0u;
        }
        for (uint i = total64 + t; i < (uint)MAX64; i += (uint)TG) {
            tiles64[2 * i] = 0u;
            tiles64[2 * i + 1] = 0u;
        }
        """#

}
