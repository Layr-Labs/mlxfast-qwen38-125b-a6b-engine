// TrackPleRowCache.swift -- a dequantized-row cache over the n-gram row source.
//
// WHAT EVERY GATHER PAYS TODAY. `Qwen4ExpNGramTable` keeps the 29.8 GiB
// n-gram table mapped MADV_RANDOM behind a bounded LRU of RAW rows. A warm
// gather is still real work on every decode step: the gather lock, the
// dedup walk, sixteen row memcpys into three fresh Data buffers, three
// host->device array constructions, and the dequantize graph build -- all
// before the PLE block sees a single value, and all repeated verbatim when
// the same trigram context comes up again.
//
// WHAT THIS ADDS. The row-source contract is that `rows` returns the same
// values for the same ids on every call, so the DEQUANTIZED result is
// cacheable at the seam the runner already owns. `TrackPleRowCacheSource`
// wraps the source the loader resolves: a hit returns the exact array a
// previous call produced -- no lock, no memcpy, no allocations, no
// dequantize graph. The trigram contexts the ids hash are heavily skewed,
// so the hot set is small and stable inside one decode.
//
// EXACTNESS. A hit returns the same MLXArray the inner source produced for
// the same (ids, shape) -- the same dequantized checkpoint rows in the same
// order, bit for bit. Nothing is recomputed, re-rounded, or re-ordered.
// MLXArray values are immutable; sharing one instance across calls changes
// no value any consumer reads. A miss delegates to the inner source and is
// byte-identical to the unwrapped path.
//
// SCOPE. Only small gathers are cached: the decode and verify windows the
// fast path serves (S <= 8 -> <= 128 ids). Prefill-width gathers pass
// through to the inner source exactly as before -- their key hashing would
// cost more than the gather saves and their entries would be megabytes.

import Foundation
import MLX
import MLXLLM

final class TrackPleRowCacheSource: Qwen4ExpNGramHostRowSource {
    private struct Key: Hashable {
        let ids: [Int]
        let shape: [Int]
    }

    /// Largest id list worth caching: the S <= 8 host-path windows produce
    /// at most 8 * 16 = 128 ids per call.
    private static let maxCachedIds = 128
    /// Resident entries before FIFO eviction. One decode entry is
    /// ~16 rows x ~160 dims x 2 B, so 32768 entries bound the cache near
    /// 160 MiB.
    private static let capacity = 32768

    private let host: any Qwen4ExpNGramHostRowSource
    private let lock = NSLock()
    private var rows: [Key: MLXArray] = [:]
    /// Insertion-order ring for FIFO eviction; O(1) per insert.
    private var ring: [Key] = []
    private var head = 0

    var rowDimensions: Int { host.rowDimensions }

    init(host: any Qwen4ExpNGramHostRowSource) {
        self.host = host
    }

    /// The device-id form: the same conversion the table itself performs,
    /// then the cached host path.
    func rows(globalIds: MLXArray) -> MLXArray {
        rows(
            globalIds: globalIds.asType(.int32).asArray(Int32.self).map(Int.init),
            shape: globalIds.shape)
    }

    /// The host-id form the fast model calls every decode step.
    func rows(globalIds ids: [Int], shape: [Int]) -> MLXArray {
        guard ids.count <= Self.maxCachedIds else {
            return host.rows(globalIds: ids, shape: shape)
        }

        let key = Key(ids: ids, shape: shape)
        lock.lock()
        if let hit = rows[key] {
            lock.unlock()
            return hit
        }
        lock.unlock()

        let produced = host.rows(globalIds: ids, shape: shape)

        lock.lock()
        if rows[key] == nil {
            rows[key] = produced
            if ring.count < Self.capacity {
                ring.append(key)
            } else {
                rows.removeValue(forKey: ring[head])
                ring[head] = key
                head = (head &+ 1) % Self.capacity
            }
        }
        lock.unlock()
        return produced
    }
}
