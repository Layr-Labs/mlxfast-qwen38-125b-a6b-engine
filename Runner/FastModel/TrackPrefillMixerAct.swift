import Foundation
import MLX
import MLXNN

enum TrackPrefillMixerAct {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_MIXER_SILU_EPILOGUE"] != "0"

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

    /// M-tile for the mixer-act down projection.
    ///
    /// MLXFAST-MIXACTBM: the down projection is 1024x320x10240. BN, BK, WM, WN
    /// stay 64, 64, 2, 2 — the K traversal (`for k = 0; k < K; k += BK`) and
    /// the N-tile are untouched. Only BM changes which rows share a threadgroup
    /// and the per-thread register tile `TM = (BM / WM) / 16`.
    ///
    /// Occupancy on a 40-core GPU, 128 threads/TG, shm 9216 B (BN × 72 × 2;
    /// independent of BM, under the 32 KiB budget):
    ///
    ///   BM  TM  TGs  thr/core  notes
    ///   32   1  160       512  tip form (default; unset == 711083e)
    ///   64   2   80       256  old tip b4646c9 (opt-in via MLXFAST_MIXACT_TILE=64)
    ///   16   0  320      1024  REJECT: empty register tile
    ///
    /// BM=16 with WM=1 recovers TM=1, but the weight loader requires
    /// `(BCOLS_PACKED / n_reads) == n_groups` → n_reads=16 → tgp=128 →
    /// WM×WN=4. The pair that keeps tgp=128 and TM≥1 is WM=1, WN=4, which
    /// drops TN from 2 to 1 (register tile not flat) and changes N-direction
    /// simdgroups. Out of scope. Env "16" falls through to the default.
    struct MTile: Equatable, Sendable {
        let bm: Int
        let wm: Int
        let wn: Int

        static let simdSize = 32
        static let bn = 64
        static let bk = 64
        static let n = 320
        static let productionRows = 1024
        static let gpuCores = 40
        /// bf16 BK pad: BK + 16/sizeof(bf16) = 72.
        static let bkPaddedBf16 = 72
        static let weightShmBytes = bn * bkPaddedBf16 * MemoryLayout<UInt16>.size

        var sm: Int { bm / wm }
        var tm: Int { sm / 16 }
        var threadgroupThreads: Int { Self.simdSize * wm * wn }
        var shipped: Bool { tm >= 1 && threadgroupThreads == 128 && (bm == 64 || bm == 32) }

        func mTiles(rows: Int) -> Int { (rows + bm - 1) / bm }
        func nTiles() -> Int { (Self.n + Self.bn - 1) / Self.bn }
        func threadgroups(rows: Int) -> Int { mTiles(rows: rows) * nTiles() }
        func grid(rows: Int) -> (Int, Int, Int) {
            (nTiles() * Self.simdSize, mTiles(rows: rows) * wm, wn)
        }
        var threadGroup: (Int, Int, Int) { (Self.simdSize, wm, wn) }
    }

    static let tile64 = MTile(bm: 64, wm: 2, wn: 2)
    static let tile32 = MTile(bm: 32, wm: 2, wn: 2)
    static let tile16Rejected = MTile(bm: 16, wm: 2, wn: 2)
    static let shippedTiles = [tile64, tile32]
    static let defaultTile = tile32

    static func tile(fromEnv value: String?) -> MTile {
        switch value {
        case "64": return tile64
        case "32": return tile32
        default: return defaultTile
        }
    }

    static var selectedTile: MTile {
        tile(fromEnv: ProcessInfo.processInfo.environment["MLXFAST_MIXACT_TILE"])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_mixer_silu_epilogue",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: source,
        header: TrackPrefillIndirect.metalHeader + TrackFastKernels.exactHeader + denseHeader,
        ensureRowContiguous: true)

    static func apply(_ projection: TrackProj, x: MLXArray, width: Int) -> MLXArray? {
        guard supportsNAX else { return nil }
        return apply(projection, x: x, width: width, tile: selectedTile)
    }

    /// Launch one shipped M-tile. Production `apply` still requires NAX.
    static func apply(_ projection: TrackProj, x: MLXArray, width: Int, tile: MTile)
        -> MLXArray?
    {
        guard enabled, StreamOrDevice.default.stream == Stream.gpu,
            tile.shipped, width == 320, x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 1024,
            x.dim(2) == 10240, x.dtype == .bfloat16,
            case .quant(let q) = projection, q.groupSize == 32, q.bits == 4,
            q.mode == .affine, q.weight.shape == [320, 1280], q.weight.dtype == .uint32,
            q.scales.shape == [320, 320], q.scales.dtype == .bfloat16,
            let biases = q.biases, biases.shape == q.scales.shape, biases.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(1)
        return kernel(
            [x, q.weight, q.scales, biases],
            template: [("T", x.dtype), ("M", rows), ("BM", tile.bm)],
            grid: tile.grid(rows: rows), threadGroup: tile.threadGroup,
            outputShapes: [[1, rows, 320]], outputDTypes: [.bfloat16])[0]
    }

    /// MLXFAST-MIXACTBM: the down projection is 1024x320x10240, so the stock
    /// 64-row M tile leaves only 16 M tiles x 5 N tiles = 80 threadgroups for a
    /// 40-core GPU. A 32-row M tile doubles that to 160. The K traversal is
    /// untouched: `for (k = 0; k < K; k += BK)` with the same BK = 64 and the
    /// same `kk1` MMAs inside, so every output element still accumulates its K
    /// blocks in the same order -- only which rows share a threadgroup changes.
    /// `TM = SM / 16` becomes 1 instead of 2 and the register tile halves.
    static let source = #"""
        alignas(16) threadgroup T Ws[64 * 72];
        track_mixer_act_dense<T, 32, 4, true, BM, 64, 64, 2, 2>(
            w, scales, biases, x, y, Ws, 10240, 320, M,
            threadgroup_position_in_grid, thread_index_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
