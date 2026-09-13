// TrackPLEWarm.swift -- untimed PLE n-gram LRU page warm at adopt.
//
// The n-gram table is 128 split parts (384 tensors) behind a bounded LRU.
// The first scored window pays SSD faults for that working set. The set is
// a hash of token history, so it is not known at load. This pass therefore
// touches a fixed resident set through the host LRU gather, once, during
// untimed adopt:
//
//   * row 0 of every split part (the root page of each shard)
//   * extra rows of the first 8 split parts
//   * cap at Qwen4ExpNGramTable.hotPathMaximumRows (4096) so the gather
//     stays on the LRU path, not the prefill mmap/pread path
//
// Arithmetic on the pinned table (128 parts, 2_500_012 rows each):
//   8 * (1 + 496) + (128 - 8) * 1 = 3976 + 120 = 4096.
//
// TRACK_PLE_WARM=0 skips the pass. A second adopt is a no-op. The default
// stream must be the GPU stream; a model-holding GPU run that is already
// past adopt is unaffected because the pass is done. Dummy gather arrays
// are dropped with Memory.clearCache() after the host read.

import Foundation
import MLX
import MLXLLM

protocol TrackPLEWarmSource: AnyObject {
    func touchThroughLRU(_ ids: [Int])
}

enum TrackPLEWarm {
    static let firstSplitParts = 8
    static let extraRowsPerPart = 496

    enum Outcome: Equatable {
        case warmed
        case skippedToggleOff
        case skippedAlreadyWarmed
        case skippedNotGPU
        case skippedNoSource
    }

    struct Receipt: Equatable {
        let outcome: Outcome
        let plannedIds: [Int]
        let touchedIds: [Int]
    }

    /// Process-wide adopt guard. Tests pass a fresh `State` so they do not
    /// share this flag.
    final class State: @unchecked Sendable {
        fileprivate let lock = NSLock()
        fileprivate var done = false
    }

    private static let processState = State()

    static func enabledValue(_ raw: String?) -> Bool {
        (raw ?? "1") != "0"
    }

    static func isEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        enabledValue(environment["TRACK_PLE_WARM"])
    }

    /// Row ids the warm pass will gather. Order is first N parts, then the
    /// remaining parts' row 0. Unique, and never longer than the LRU hot path.
    static func intendedRowIds(layout: Qwen4ExpNGramTableLayout) -> [Int] {
        guard layout.shardCount > 0, layout.rowsPerShard > 0 else { return [] }
        let cap = Qwen4ExpNGramTable.hotPathMaximumRows
        let parts = min(firstSplitParts, layout.shardCount)
        let extra = min(extraRowsPerPart, max(0, layout.rowsPerShard - 1))
        var ids: [Int] = []
        ids.reserveCapacity(min(cap, layout.rowCount))
        for shard in 0 ..< parts {
            let base = shard * layout.rowsPerShard
            let take = min(1 + extra, layout.rowsPerShard)
            for row in 0 ..< take {
                ids.append(base + row)
                if ids.count == cap { return ids }
            }
        }
        if parts < layout.shardCount {
            for shard in parts ..< layout.shardCount {
                ids.append(shard * layout.rowsPerShard)
                if ids.count == cap { return ids }
            }
        }
        return ids
    }

    @discardableResult
    static func warm(
        source: TrackPLEWarmSource,
        layout: Qwen4ExpNGramTableLayout,
        enabled: Bool = isEnabled(),
        streamIsGPU: Bool = (StreamOrDevice.default.stream === Stream.gpu),
        state: State = processState
    ) -> Receipt {
        let planned = intendedRowIds(layout: layout)
        guard enabled else {
            return Receipt(outcome: .skippedToggleOff, plannedIds: planned, touchedIds: [])
        }
        guard streamIsGPU else {
            return Receipt(outcome: .skippedNotGPU, plannedIds: planned, touchedIds: [])
        }
        state.lock.lock()
        if state.done {
            state.lock.unlock()
            return Receipt(outcome: .skippedAlreadyWarmed, plannedIds: planned, touchedIds: [])
        }
        state.done = true
        state.lock.unlock()
        guard !planned.isEmpty else {
            return Receipt(outcome: .warmed, plannedIds: planned, touchedIds: [])
        }
        source.touchThroughLRU(planned)
        return Receipt(outcome: .warmed, plannedIds: planned, touchedIds: planned)
    }

    @discardableResult
    static func warm(model: Qwen4ExpModel) -> Receipt {
        guard let embedding = model.pleEmbeddings.first,
            let source = embedding.rowSourceHolder.source
        else {
            return Receipt(outcome: .skippedNoSource, plannedIds: [], touchedIds: [])
        }
        let layout = Qwen4ExpNGramTableLayout(
            shardCount: embedding.shardCount,
            rowsPerShard: embedding.rowsPerShard,
            rowDimensions: embedding.rowDimensions)
        if let host = source as? Qwen4ExpNGramHostRowSource {
            return warm(source: HostRowWarmTarget(host: host), layout: layout)
        }
        return warm(source: DeviceRowWarmTarget(source: source), layout: layout)
    }
}

/// Public LRU gather: `rows(globalIds:shape:)` copies through the hot-row
/// cache. The returned arrays are dropped; the point is the host read.
private final class HostRowWarmTarget: TrackPLEWarmSource {
    let host: Qwen4ExpNGramHostRowSource
    init(host: Qwen4ExpNGramHostRowSource) { self.host = host }
    func touchThroughLRU(_ ids: [Int]) {
        _ = host.rows(globalIds: ids, shape: [ids.count])
        Memory.clearCache()
    }
}

private final class DeviceRowWarmTarget: TrackPLEWarmSource {
    let source: Qwen4ExpNGramRowSource
    init(source: Qwen4ExpNGramRowSource) { self.source = source }
    func touchThroughLRU(_ ids: [Int]) {
        _ = source.rows(globalIds: MLXArray(ids.map { Int32($0) }))
        Memory.clearCache()
    }
}
