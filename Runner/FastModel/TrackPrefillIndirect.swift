import Foundation
import MLX
import MLXLMCommon

enum TrackPrefillIndirect {
    /// Gate|up weight-block (BN × BK). BM=32, WM=WN=2, 128 threads, T=bf16.
    /// Two weight stagings (Ws0, Ws1) plus one A staging.
    /// `BK_padded = BK + 16/sizeof(T)` matches the Metal loader.
    ///
    /// `MLXFAST_GU_TILE` is accepted but selects nothing: `64x64` is the
    /// only accepted tile, so unset, unrecognized, and `128x32` all resolve
    /// to the tip form. 128×32 moved to `rejectedGUTiles` — the frontier
    /// 6fbe0e4 MLXFAST-ACC rewrite of the GU kernel implements the TN=2
    /// walk only, and 128×32 is TN=4. Rejected BN∈{64,128} × BK∈{32,64}
    /// candidates stay in `rejectedGUTiles`.
    ///
    /// K-order: SK=32 and BK % SK == 0 on every accepted tile, so the walk
    /// `for k in 0..<K/BK { for kk1 in 0..<BK stride SK { mma } }` adds K
    /// as 32-wide blocks 0, 32, …, 2528 — the same sequence as 64×64.
    struct GUTile: Equatable, Sendable {
        let blockN: Int
        let blockK: Int

        var name: String { "\(blockN)x\(blockK)" }

        static let bm = 32
        static let threadsPerThreadgroup = 128
        static let elementBytes = 2
        static let bankPad = 16 / elementBytes
        /// Tip GU tile (64×64, two weight banks). 32 KB is the API cap.
        static let existingBudgetBytes = 23_040
        static let apiCapBytes = 32 * 1024
        /// Apple7/8 physical threadgroup memory per core (PB-607).
        static let physicalBytesPerCore = 60 * 1024
        static let currentResident = 2

        var bkPadded: Int { blockK + Self.bankPad }
        var threadgroupBytes: Int {
            (2 * blockN + Self.bm) * bkPadded * Self.elementBytes
        }
        var threadsPerThreadgroup: Int { Self.threadsPerThreadgroup }
        /// PackedNAXGroup32 requires n_reads==16 ⇒ BN×BK==4096.
        var packedReads: Int { (blockK / 2 * blockN) / Self.threadsPerThreadgroup }
        var naxTN: Int { blockN / 32 }
        var residentThreadgroupsPerCore: Int {
            Self.physicalBytesPerCore / max(threadgroupBytes, 1)
        }
        var fitsAPICap: Bool { threadgroupBytes <= Self.apiCapBytes }
        var fitsExistingBudget: Bool { threadgroupBytes <= Self.existingBudgetBytes }
        var threadGroup: (Int, Int, Int) { (32, 2, 2) }
        func grid(maxTiles: Int, n: Int) -> (Int, Int, Int) {
            ((n / blockN) * 32, maxTiles * 2, 2)
        }
    }

    struct RejectedGUTile: Sendable {
        let tile: GUTile
        let reason: String
    }

    /// P17 down weight-block (BN × BK). BM=32, WM=WN=2, 128 threads, T=bf16.
    /// One weight staging plus A. `BK_padded = BK + 16/sizeof(T)` matches
    /// the Metal loader.
    ///
    /// `MLXFAST_P17_TILE` selects an accepted down tile (`64x64`, `128x32`).
    /// Unset or unrecognized → `128x32`, the current winner.
    /// `MLXFAST_DOWNBLOCK_N` selects BN. BK is the PackedNAX-legal partner
    /// (BN×BK == 4096). Unset or unrecognized → 128 (BK=32), the tip form.
    /// 256 is rejected. The M-tile table is BM=32 on both sides and does not
    /// read this knob; host grid X and kernel BN both read the selected tile
    /// so the N-tile producer matches the consumer.
    ///
    /// K-order: SK=32 and BK % SK == 0 on every accepted tile, so the walk
    /// adds K as 32-wide blocks 0, 32, …, 608 — the same sequence as 64×64.
    struct DownTile: Equatable, Sendable {
        let blockN: Int
        let blockK: Int

        var name: String { "\(blockN)x\(blockK)" }

        static let bm = 32
        static let threadsPerThreadgroup = 128
        static let elementBytes = 2
        static let bankPad = 16 / elementBytes
        /// Larger of the two shipping tiles (64×64 oracle). 32 KB is the API cap.
        static let existingBudgetBytes = 13_824
        static let apiCapBytes = 32 * 1024
        /// Apple7/8 physical threadgroup memory per core (PB-607).
        static let physicalBytesPerCore = 60 * 1024
        static let currentResident = 4

        var bkPadded: Int { blockK + Self.bankPad }
        var threadgroupBytes: Int {
            (blockN + Self.bm) * bkPadded * Self.elementBytes
        }
        var threadsPerThreadgroup: Int { Self.threadsPerThreadgroup }
        /// PackedNAXGroup32 requires n_reads==16 ⇒ BN×BK==4096.
        var packedReads: Int { (blockK / 2 * blockN) / Self.threadsPerThreadgroup }
        var naxTN: Int { blockN / 32 }
        var residentThreadgroupsPerCore: Int {
            Self.physicalBytesPerCore / max(threadgroupBytes, 1)
        }
        var fitsAPICap: Bool { threadgroupBytes <= Self.apiCapBytes }
        var fitsExistingBudget: Bool { threadgroupBytes <= Self.existingBudgetBytes }
        var threadGroup: (Int, Int, Int) { (32, 2, 2) }
        func grid(maxTiles: Int, n: Int) -> (Int, Int, Int) {
            ((n / blockN) * 32, maxTiles * 2, 2)
        }
    }

    struct RejectedDownTile: Sendable {
        let tile: DownTile
        let reason: String
    }

    static let defaultGUTile = GUTile(blockN: 64, blockK: 64)

    static let acceptedGUTiles: [GUTile] = [
        GUTile(blockN: 64, blockK: 64),
    ]

    static let rejectedGUTiles: [RejectedGUTile] = [
        RejectedGUTile(
            tile: GUTile(blockN: 64, blockK: 32),
            reason: "n_reads=8 but PackedNAXGroup32 requires 16; loader (BCOLS_PACKED/n_reads)==n_groups fails (2!=1)"
        ),
        RejectedGUTile(
            tile: GUTile(blockN: 128, blockK: 32),
            reason: "frontier 6fbe0e4 MLXFAST-ACC GU walk implements TN=2 only (frag_at(0)/(1), val_frags[0]/[1]); 128x32 is TN=4 and half the output columns stay clear() zeros"
        ),
        RejectedGUTile(
            tile: GUTile(blockN: 128, blockK: 64),
            reason: "threadgroup 41472 B exceeds existing 23040 B budget and the 32 KB API cap; n_reads=32 but PackedNAXGroup32 requires 16; occupancy 1 tg/core vs 2"
        ),
    ]

    static func guTile(named raw: String?) -> GUTile {
        guard let raw, let tile = acceptedGUTiles.first(where: { $0.name == raw }) else {
            return defaultGUTile
        }
        return tile
    }

    /// `MLXFAST_P17_TILE` selects an accepted down tile (`64x64`, `128x32`).
    /// Unset or unrecognized → `128x32`, the current winner. Rejected
    /// candidates stay in `rejectedDownTiles` with the budget/loader math.
    /// `MLXFAST_DOWNBLOCK_N` selects BN (`64`, `128`). Unset or unrecognized
    /// → `128x32`. 256 stays in `rejectedDownTiles`.
    static let defaultDownTile = DownTile(blockN: 128, blockK: 32)

    static let acceptedDownTiles: [DownTile] = [
        DownTile(blockN: 64, blockK: 64),
        DownTile(blockN: 128, blockK: 32),
    ]

    static let rejectedDownTiles: [RejectedDownTile] = [
        RejectedDownTile(
            tile: DownTile(blockN: 256, blockK: 16),
            reason: "BK=16 is not a multiple of group_size=32; PackedNAX store writes 32 K; SK=32 would overrun the K tile (changing SK would change the K-add order)"
        ),
        RejectedDownTile(
            tile: DownTile(blockN: 128, blockK: 64),
            reason: "threadgroup 23040 B exceeds existing 13824 B budget; n_reads=32 but PackedNAXGroup32 requires 16; occupancy 2 tg/core vs 4"
        ),
        RejectedDownTile(
            tile: DownTile(blockN: 256, blockK: 32),
            reason: "threadgroup 23040 B exceeds existing 13824 B budget; n_reads=32 but PackedNAXGroup32 requires 16; occupancy 2 tg/core vs 4"
        ),
    ]

    static func downTile(named raw: String?) -> DownTile {
        guard let raw, let tile = acceptedDownTiles.first(where: { $0.name == raw }) else {
            return defaultDownTile
        }
        return tile
    }

    static func downBlock(fromEnv raw: String?) -> DownTile {
        switch raw {
        case "64": return DownTile(blockN: 64, blockK: 64)
        case "128": return DownTile(blockN: 128, blockK: 32)
        default: return defaultDownTile
        }
    }

    static func downSource(tile: DownTile) -> String {
        tile.blockN == downBlockN ? sourceDown : source
    }

    /// Prefer `MLXFAST_DOWNBLOCK_N` when set; otherwise `MLXFAST_P17_TILE`.
    /// Both unset → 128×32, the tip form of either knob.
    private static let selectedDownTile: DownTile = {
        if ProcessInfo.processInfo.environment["MLXFAST_DOWNBLOCK_N"] != nil {
            return downBlock(fromEnv: ProcessInfo.processInfo.environment["MLXFAST_DOWNBLOCK_N"])
        }
        return downTile(named: ProcessInfo.processInfo.environment["MLXFAST_P17_TILE"])
    }()

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
    /// Unset `MLXFAST_P17_TILE` and unset `MLXFAST_DOWNBLOCK_N` keep this
    /// kernel and this grid.
    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_indirect_down",
        inputNames: ["x", "w", "scales", "biases", "indices", "token_rows", "tiles"],
        outputNames: ["y"], source: sourceDown, header: metalHeader,
        ensureRowContiguous: true)

    private static let kernel64x64 = MLXFast.metalKernel(
        name: "track_prefill_indirect_down_64x64",
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

    static func apply(
        _ m: TrackMoE, x: MLXArray, indices: MLXArray, dispatch: TrackMoEDispatch.Pack? = nil
    ) -> (activated: MLXArray, sortedIDs: MLXArray, inverse: MLXArray, tiles: MLXArray)? {
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

        let sortedIDs: MLXArray
        let inverse: MLXArray
        let tokenRows: MLXArray
        if let dispatch {
            (sortedIDs, tokenRows, inverse) = (dispatch.sortedIDs, dispatch.tokenRows, dispatch.inverse)
        } else if let c = TrackPrefillSort.apply(
            flatIDs: indices.flattened(), experts: g.w.dim(0), topK: indices.dim(2))
        {
            // The identical permutation in two launches (see TrackPrefillSort).
            (sortedIDs, tokenRows, inverse) = (c.sortedIDs, c.tokenRows, c.inverse)
        } else {
            let flatIDs = indices.flattened()
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
        let tile = selectedDownTile
        if tile.blockN == downBlockN {
            return kernel(
                [activated, d.w, d.s, d.b, sortedIDs, identityRows(rows), tiles],
                template: [("T", activated.dtype), ("N", 2560), ("K", 640)],
                grid: ((2560 / downBlockN) * 32, maxT * 2, 2),
                threadGroup: (32, 2, 2),
                outputShapes: [[rows, 1, 2560]], outputDTypes: [.bfloat16])[0]
        }
        return down(
            tile: tile, activated: activated, w: d.w, scales: d.s, biases: d.b,
            sortedIDs: sortedIDs, tiles: tiles)
    }

    /// Launch the gate|up kernel at one accepted tile (64×64 is the only
    /// one, so `tile` selects nothing today; it stays for the sweep oracle).
    static func gateUp(
        tile: GUTile,
        x: MLXArray,
        w0: MLXArray,
        scales0: MLXArray,
        biases0: MLXArray,
        w1: MLXArray,
        scales1: MLXArray,
        biases1: MLXArray,
        sortedIDs: MLXArray,
        tokenRows: MLXArray,
        tiles: MLXArray
    ) -> MLXArray {
        let rows = sortedIDs.size
        let n = 640
        let maxT = maxTiles(rows: rows, experts: w0.dim(0))
        return gateUpKernel(
            [x, w0, scales0, biases0, w1, scales1, biases1, sortedIDs, tokenRows, tiles],
            template: [("T", x.dtype), ("N", n), ("K", 2560), ("SILU", true)],
            grid: tile.grid(maxTiles: maxT, n: n),
            threadGroup: tile.threadGroup,
            outputShapes: [[rows, 1, n]], outputDTypes: [.bfloat16])[0]
    }

    /// Launch one accepted down tile. The default 128×32 path in `down(_:activated:…)`
    /// does not go through here.
    static func down(
        tile: DownTile,
        activated: MLXArray,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        sortedIDs: MLXArray,
        tiles: MLXArray
    ) -> MLXArray {
        let rows = sortedIDs.size
        let n = 2560
        let maxT = maxTiles(rows: rows, experts: w.dim(0))
        let compiled = tile.blockN == downBlockN ? kernel : kernel64x64
        return compiled(
            [activated, w, scales, biases, sortedIDs, identityRows(rows), tiles],
            template: [("T", activated.dtype), ("N", n), ("K", 640)],
            grid: tile.grid(maxTiles: maxT, n: n),
            threadGroup: tile.threadGroup,
            outputShapes: [[rows, 1, n]], outputDTypes: [.bfloat16])[0]
    }

    nonisolated(unsafe) private static var identityCache: [Int: MLXArray] = [:]
    nonisolated(unsafe) private static let identityCacheLock = NSLock()

    /// `0 ..< rows` as uint32, built once per row count. Lock-guarded: the
    /// cache is `nonisolated(unsafe)` because callers arrive off the main
    /// actor, and two concurrent `down` calls once raced the dictionary
    /// (parallel test execution corrupted the buckets and trapped in
    /// `-[NSTaggedPointerString count]`).
    private static func identityRows(_ rows: Int) -> MLXArray {
        identityCacheLock.lock()
        defer { identityCacheLock.unlock() }
        if let cached = identityCache[rows] { return cached }
        let a = MLXArray((0..<rows).map { UInt32($0) })
        eval(a)
        identityCache[rows] = a
        return a
    }

    static let sourceGU = #"""
        alignas(16) threadgroup T Ws0[64 * 72];
        alignas(16) threadgroup T Ws1[64 * 72];
        alignas(16) threadgroup T As[32 * 72];
        track_prefill_indirect_gu<T, 32, 4, 32, 64, 64, 2, 2, true, SILU, N>(
            x, w0, scales0, biases0, w1, scales1, biases1, indices, token_rows, tiles,
            y0, y1, N, K, Ws0, Ws1, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

    /// Default BN. Unset `MLXFAST_P17_TILE` / `MLXFAST_DOWNBLOCK_N` keeps 128.
    static let downBlockN = 128

    static let sourceDown = #"""
        alignas(16) threadgroup T Ws[128 * 40];
        alignas(16) threadgroup T As[32 * 40];
        track_prefill_indirect<T, 32, 4, 32, 128, 32, 2, 2, true, N>(
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
}
