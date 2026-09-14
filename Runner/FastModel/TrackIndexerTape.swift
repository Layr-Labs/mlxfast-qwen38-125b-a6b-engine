// TrackIndexerTape.swift -- capacity-fixed indexer-key buffer for the fast path.
//
// Vendor `Qwen4ExpCBv2LayerCache.updateIndexerTape` (and the head's
// `Qwen4ExpAttentionCache.updateIndexer`) concatenates an exact-length tape
// every token. Both types are `final` with a private store, so the concat
// cannot be redirected from this module. While the fast path owns the
// forward, keep-mask is unused (context stays at or under the indexer
// budget) and the returned tape is discarded. Write each step's keys at
// `offset` into a buffer allocated once at that budget, then on vendor
// fallback copy the live prefix in with one `updateIndexerTape` /
// `updateIndexer` so the keep-mask consumer sees the same rows.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

enum TrackIndexerTape {
    /// `TRACK_INDEXER_TAPE_RING=0` restores the per-token vendor concat.
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_INDEXER_TAPE_RING"] ?? "1") != "0"
    }()

    private struct Slot {
        var generation: ObjectIdentifier
        var buffer: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var slots: [ObjectIdentifier: Slot] = [:]

    static func liveBuffers() -> [MLXArray] {
        lock.lock()
        defer { lock.unlock() }
        return slots.map(\.value.buffer)
    }

    /// Append `keys` `[B, S, D]` at the row's pre-update offset, or fall
    /// through to the vendor concat when the vendor tape already exists.
    static func appendCBv2(
        cache: Qwen4ExpCBv2LayerCache, keys: MLXArray, capacity: Int
    ) {
        if enabled, cache.indexerTapeLength == 0, cache.rows.count == 1 {
            let row = cache.rows[0]
            write(
                owner: cache, generation: ObjectIdentifier(row),
                keys: keys, offset: row.absoluteOffset, capacity: capacity)
        } else {
            _ = cache.updateIndexerTape(keys: keys)
        }
    }

    /// Copy the live prefix into an empty vendor tape so a keep-mask fallback
    /// concatenates only this step, not a missing history.
    static func syncCBv2(_ caches: [Qwen4ExpCBv2LayerCache]) {
        guard enabled else { return }
        for cache in caches {
            guard cache.indexerTapeLength == 0, cache.rows.count == 1 else { continue }
            let committed = cache.rows[0].absoluteOffset
            guard let prefix = prefix(owner: cache, count: committed) else { continue }
            _ = cache.updateIndexerTape(keys: prefix)
        }
    }

    static func appendHead(
        cache: Qwen4ExpAttentionCache, keys: MLXArray, capacity: Int
    ) {
        if enabled, cache.indexerKeys == nil {
            write(
                owner: cache, generation: ObjectIdentifier(cache),
                keys: keys, offset: cache.offset, capacity: capacity)
        } else {
            _ = cache.updateIndexer(keys: keys)
        }
    }

    static func syncHead(_ cache: Qwen4ExpAttentionCache) {
        guard enabled, cache.indexerKeys == nil else { return }
        guard let prefix = prefix(owner: cache, count: cache.offset) else { return }
        _ = cache.updateIndexer(keys: prefix)
    }

    static func buffer(owner: AnyObject) -> MLXArray? {
        lock.lock()
        defer { lock.unlock() }
        return slots[ObjectIdentifier(owner)]?.buffer
    }

    private static func write(
        owner: AnyObject, generation: ObjectIdentifier,
        keys: MLXArray, offset: Int, capacity: Int
    ) {
        let S = keys.dim(1)
        precondition(
            offset >= 0 && S >= 0 && offset + S <= capacity,
            "TrackIndexerTape: write [\(offset), \(offset + S)) exceeds capacity \(capacity)")
        let buf = buffer(
            owner: owner, generation: generation, keys: keys, capacity: capacity)
        // Axis-1 slice update, same primitive as CompilableKVCache. The Swift
        // subscript trims leading singleton dims, which would collapse decode
        // keys `[1, 1, D]` to `[D]`.
        buf._updateInternal(
            dynamicSliceUpdate(
                buf, update: keys, start: MLXArray([Int32(offset)]), axes: [1]))
    }

    private static func prefix(owner: AnyObject, count: Int) -> MLXArray? {
        guard count > 0 else { return nil }
        lock.lock()
        let buf = slots[ObjectIdentifier(owner)]?.buffer
        lock.unlock()
        guard let buf, buf.dim(1) >= count else { return nil }
        return contiguous(buf[0..., ..<count, 0...])
    }

    private static func buffer(
        owner: AnyObject, generation: ObjectIdentifier,
        keys: MLXArray, capacity: Int
    ) -> MLXArray {
        let id = ObjectIdentifier(owner)
        lock.lock()
        if let slot = slots[id],
            slot.generation == generation,
            slot.buffer.dim(0) == keys.dim(0),
            slot.buffer.dim(2) == keys.dim(2),
            slot.buffer.dtype == keys.dtype
        {
            let existing = slot.buffer
            lock.unlock()
            return existing
        }
        lock.unlock()
        let buf = MLXArray.zeros(
            [keys.dim(0), capacity, keys.dim(2)], dtype: keys.dtype)
        eval(buf)
        lock.lock()
        slots[id] = Slot(generation: generation, buffer: buf)
        lock.unlock()
        return buf
    }
}
