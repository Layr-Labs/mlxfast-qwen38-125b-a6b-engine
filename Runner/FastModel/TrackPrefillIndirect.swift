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

    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_indirect_activations",
        inputNames: ["x", "w", "scales", "biases", "indices", "token_rows", "tiles"],
        outputNames: ["y"], source: source, header: metalHeader,
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
        -> (activated: MLXArray, sortedIDs: MLXArray, inverse: MLXArray)?
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
        func project(_ bank: (w: MLXArray, s: MLXArray, b: MLXArray)) -> MLXArray {
            kernel(
                [x, bank.w, bank.s, bank.b, sortedIDs, tokenRows, tiles],
                template: [("T", x.dtype), ("N", 640), ("K", 2560)],
                grid: (10 * 32, maxT * 2, 2),
                threadGroup: (32, 2, 2),
                outputShapes: [[rows, 1, 640]], outputDTypes: [.bfloat16])[0]
        }
        let up = project(u)
        let gate = project(g)
        return (compiledSiluProduct(gate, up), sortedIDs, inverse)
    }

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
