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

<<<<<<< Updated upstream
    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_indirect_activations",
        inputNames: ["x", "w", "scales", "biases", "indices", "token_rows", "tiles"],
        outputNames: ["y"], source: source, header: metalHeader,
=======
    /// Down projection: 128x32 weight blocks (20 column tiles over N = 2560,
    /// 20 K steps over K = 640); the 64x64 block `source` remains the reference.
    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_indirect_down",
        inputNames: ["x", "w", "scales", "biases", "indices", "token_rows", "tiles"],
        outputNames: ["y"], source: sourceDown, header: metalHeader,
>>>>>>> Stashed changes
        ensureRowContiguous: true)

    /// `[2 * maxTiles]` of `[begin, end)` row ranges, one 32-row tile per slot,
    /// each aligned to the start of its expert run; unused slots are `[0, 0)`.
    private static let tileKernel = MLXFast.metalKernel(
        name: "track_prefill_tile_table",
        inputNames: ["sorted_ids"], outputNames: ["tiles"],
        source: tileSource, header: "", ensureRowContiguous: true)

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
        -> (activated: MLXArray, sortedIDs: MLXArray, inverse: MLXArray, tiles: MLXArray)?
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
        let tiles = tileTable(sortedIDs: sortedIDs, rows: rows, experts: experts)
        let maxT = maxTiles(rows: rows, experts: experts)
        let activated = gateUpKernel(
            [x, g.w, g.s, g.b, u.w, u.s, u.b, sortedIDs, tokenRows, tiles],
            template: [("T", x.dtype), ("N", 640), ("K", 2560), ("SILU", true)],
            grid: (10 * 32, maxT * 2, 2),
            threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 640]], outputDTypes: [.bfloat16])[0]
        return (activated, sortedIDs, inverse, tiles)
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
            [activated, d.w, d.s, d.b, sortedIDs, identityRows(rows), tiles],
            template: [("T", activated.dtype), ("N", 2560), ("K", 640)],
<<<<<<< Updated upstream
            grid: (40 * 32, maxT * 2, 2),
=======
            grid: ((2560 / downBlockN) * 32, maxT * 2, 2),
>>>>>>> Stashed changes
            threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 2560]], outputDTypes: [.bfloat16])[0]
    }

    nonisolated(unsafe) private static var identityCache: [Int: MLXArray] = [:]

    /// `0 ..< rows` as uint32, built once per row count.
    private static func identityRows(_ rows: Int) -> MLXArray {
        if let cached = identityCache[rows] { return cached }
        let a = MLXArray((0..<rows).map { UInt32($0) })
        eval(a)
        identityCache[rows] = a
        return a
    }

    static let sourceGU = #"""
        threadgroup T Ws0[64 * 72];
        threadgroup T Ws1[64 * 72];
        threadgroup T As[32 * 72];
        track_prefill_indirect_gu<T, 32, 4, 32, 64, 64, 2, 2, true, SILU>(
            x, w0, scales0, biases0, w1, scales1, biases1, indices, token_rows, tiles,
            y0, y1, N, K, Ws0, Ws1, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

<<<<<<< Updated upstream
=======
    static let downBlockN = 128

    static let sourceDown = #"""
        threadgroup T Ws[128 * 40];
        threadgroup T As[32 * 40];
        track_prefill_indirect<T, 32, 4, 32, 128, 32, 2, 2, true>(
            x, w, scales, biases, indices, token_rows, tiles, y,
            N, K, Ws, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

>>>>>>> Stashed changes
    static let source = #"""
        threadgroup T Ws[64 * 72];
        threadgroup T As[32 * 72];
        track_prefill_indirect<T, 32, 4, 32, 64, 64, 2, 2, true>(
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
}
