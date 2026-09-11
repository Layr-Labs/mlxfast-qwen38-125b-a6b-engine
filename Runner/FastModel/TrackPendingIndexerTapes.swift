// MLXFAST-NOTAPE
// The short, non-capturing fast path never reads the QSA tape. Retain its
// keys until a consumer needs them, or 64 appends need to be materialized.
import MLX
import MLXLLM
import MLXLMCommon

final class TrackPendingIndexerTapes {
    private struct Entry {
        let startOffset: Int
        let keys: MLXArray
    }

    private final class Pending {
        weak var row: (any CBv2SequenceKV)?
        weak var cache: Qwen4ExpCBv2LayerCache?
        let tapeLength: Int
        var entries: [Entry] = []

        init(row: any CBv2SequenceKV, cache: Qwen4ExpCBv2LayerCache) {
            self.row = row
            self.cache = cache
            self.tapeLength = cache.indexerTapeLength
        }

        // Mirror updateIndexerTape's prefix truncation at the pre-KV-update
        // absoluteOffset, including rollback into an S > 1 pending slice.
        func truncate(to offset: Int) {
            while let last = entries.last, last.startOffset >= offset {
                entries.removeLast()
            }
            if let last = entries.last, last.startOffset + last.keys.dim(1) > offset {
                entries[entries.count - 1] = Entry(
                    startOffset: last.startOffset,
                    keys: last.keys[0..., ..<(offset - last.startOffset), 0...])
            }
        }

        var isBound: Bool {
            guard let row, let cache, cache.rows.count == 1 else { return false }
            return cache.rows[0] === row && cache.indexerTapeLength == tapeLength
        }
    }

    private var pending: [ObjectIdentifier: Pending] = [:]
    private let flushInterval = 64

    // The engine can rebind layer caches between calls. Match the real
    // cache's setRows lifetime; weak references also prevent identity reuse
    // from attaching an old request's keys to a new row or cache.
    func prune() {
        pending = pending.filter { $0.value.isBound }
    }

    func flush() {
        prune()
        for value in pending.values {
            guard let row = value.row, let cache = value.cache else { continue }
            flush(value, row: row, cache: cache)
        }
        pending.removeAll(keepingCapacity: true)
    }

    private func flush(
        _ value: Pending, row: any CBv2SequenceKV, cache: Qwen4ExpCBv2LayerCache
    ) {
        value.truncate(to: row.absoluteOffset)
        guard !value.entries.isEmpty else { return }
        let keys = value.entries.map(\.keys)
        // One cache append, rather than a growing full-tape copy per step.
        _ = cache.updateIndexerTape(
            keys: keys.count == 1 ? keys[0] : concatenated(keys, axis: 1))
    }

    // Call before updateAndAttend, just like the original eager append.
    func append(keys: MLXArray, cache: Qwen4ExpCBv2LayerCache, deferred: Bool) {
        precondition(cache.rows.count == 1)
        let row = cache.rows[0]
        let identity = ObjectIdentifier(row)
        let existing = pending[identity]
        let value: Pending
        if let existing, existing.isBound, existing.cache === cache {
            value = existing
        } else {
            value = Pending(row: row, cache: cache)
        }
        value.truncate(to: row.absoluteOffset)

        let tapeEnd = value.entries.last.map { $0.startOffset + $0.keys.dim(1) }
            ?? cache.indexerTapeLength
        if !deferred || tapeEnd != row.absoluteOffset {
            // Capture and wide windows retain the original eager behavior.
            // Rollback into materialized history must also truncate NOW:
            // after KV advances, updateIndexerTape would see a later offset.
            // A rebound cache with missing history stays eager as well; do
            // not invent absolute positions for its shorter physical tape.
            flush(value, row: row, cache: cache)
            pending.removeValue(forKey: identity)
            _ = cache.updateIndexerTape(keys: keys)
            return
        }

        value.entries.append(Entry(startOffset: row.absoluteOffset, keys: keys))
        if value.entries.count >= flushInterval {
            // The current entry is at absoluteOffset and must survive this
            // flush, so do not run the pre-append truncation a second time.
            let chunks = value.entries.map(\.keys)
            _ = cache.updateIndexerTape(keys: concatenated(chunks, axis: 1))
            pending.removeValue(forKey: identity)
        } else {
            pending[identity] = value
        }
    }
}
