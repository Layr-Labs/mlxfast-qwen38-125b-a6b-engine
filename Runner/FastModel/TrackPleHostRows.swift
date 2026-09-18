// TrackPleHostRows.swift -- Runner-resident host row path for the PLE layer.
//
// Two costs inside the vendor host path repeat every decode step even though
// nothing they read ever changes:
//
//   1. `Qwen4ExpNGramEmbedding.hostRowIds` calls
//      `constants.multipliers.asArray(Int64.self)` on EVERY invocation -- a
//      device->host readback of a 3-element array that is fixed at init. The
//      constants object is internal to MLXLLM, so the hoist reads it once
//      through Mirror and caches the integers per embedding instance.
//
//   2. `host.rows(globalIds:)` gathers and dequantizes the row set for the
//      window's n-grams. The n-gram -> row mapping is a pure function of a
//      static table (the same input independence the row source's own LRU
//      relies on), so a memo of dequantized rows keyed by the shifted n-gram
//      tuple is exact: a hit skips the hash, the LRU lookup, the copy, the
//      buffer upload and the dequant launch. Natural text repeats n-grams
//      constantly, and a verify round re-feeds context the draft steps
//      already produced.
//
// The hash below is the vendor's `hostRowIds` verbatim -- wrapping Int64
// multiply, XOR, Python-style remainder -- so a miss produces bit-identical
// global ids and the memo stores the vendor gather's own output rows.

import Foundation
import MLX
import MLXLLM

enum TrackPleHostRows {

    /// Hash inputs hoisted out of the vendor embedding, once per instance.
    private struct Constants {
        let multipliers: [Int64]
        let sizes: [Int64]
        let offsets: [Int64]
    }

    nonisolated(unsafe) private static var constants: [ObjectIdentifier: Constants] = [:]
    /// Dequantized row vector per shifted n-gram, `[1, 1, ngramHeads * headDim]`.
    nonisolated(unsafe) private static var memo: [[Int64]: MLXArray] = [:]
    private static let memoCap = 8192

    /// `constants` is internal to MLXLLM; Mirror reaches stored properties
    /// regardless of access level. Returns nil if the layout ever differs —
    /// the caller then takes the vendor path unchanged.
    private static func constants(for embedding: Qwen4ExpNGramEmbedding) -> Constants? {
        let id = ObjectIdentifier(embedding)
        if let cached = constants[id] { return cached }
        guard let constObj = Mirror(reflecting: embedding).children.first(where: {
            $0.label == "constants"
        })?.value
        else { return nil }
        var multipliers: MLXArray? = nil
        var vocabSizes: [Int]? = nil
        var headOffsets: [Int]? = nil
        for child in Mirror(reflecting: constObj).children {
            switch child.label {
            case "multipliers": multipliers = child.value as? MLXArray
            case "headVocabSizes": vocabSizes = child.value as? [Int]
            case "headOffsets": headOffsets = child.value as? [Int]
            default: break
            }
        }
        guard let multipliers, let vocabSizes, let headOffsets else { return nil }
        let extracted = Constants(
            multipliers: multipliers.asArray(Int64.self),
            sizes: vocabSizes.map(Int64.init),
            offsets: headOffsets.map(Int64.init))
        constants[id] = extracted
        return extracted
    }

    /// The window's embedded rows `[B, S, ngramHeads * headDim]` in `dtype`,
    /// or nil when the constants cannot be hoisted (caller falls back to the
    /// vendor hash + gather). `history` is `context ++ ids`, `newCount = S`.
    static func embed(
        embedding: Qwen4ExpNGramEmbedding,
        host: Qwen4ExpNGramHostRowSource,
        history: [Int64], newCount: Int,
        ngramSize: Int, headsPerNGram: Int, eosTokenId: Int,
        dtype: DType
    ) -> MLXArray? {
        guard let c = constants(for: embedding) else { return nil }
        let ngramHeads = (ngramSize - 1) * headsPerNGram
        let eos = Int64(eosTokenId)
        let T = history.count

        // previous[t] = position of the last EOS strictly before t, or -1 —
        // the vendor's segment-boundary scan, verbatim.
        var previous = [Int](repeating: -1, count: T)
        var last = -1
        for t in 0 ..< T {
            previous[t] = last
            if history[t] == eos { last = t }
        }
        func shifted(_ s: Int, _ t: Int) -> Int64 {
            if s == 0 { return history[t] }
            let inSegment = t - (previous[t] + 1)
            let source = t - s
            return (inSegment >= s && source >= 0) ? history[source] : eos
        }

        // One memo key per new position: the shifted n-gram tuple the hash
        // consumes. Rows are a pure function of it.
        var keys: [[Int64]] = []
        keys.reserveCapacity(newCount)
        var parts: [MLXArray?] = []
        parts.reserveCapacity(newCount)
        var missPositions: [Int] = []
        var missKeys: [[Int64]] = []
        for t in Swift.max(0, T - newCount) ..< T {
            var key = [Int64]()
            key.reserveCapacity(ngramSize)
            for s in 0 ..< ngramSize { key.append(shifted(s, t)) }
            keys.append(key)
            if let hit = memo[key] {
                parts.append(hit)
            } else {
                parts.append(nil)
                missPositions.append(parts.count - 1)
                missKeys.append(key)
            }
        }

        if !missKeys.isEmpty {
            // The vendor hash over the miss positions only, then ONE gather
            // for the whole miss set — same batching the inline path used.
            var gids: [Int] = []
            gids.reserveCapacity(missKeys.count * ngramHeads)
            for key in missKeys {
                for ngram in 2 ... ngramSize {
                    var mixed = key[0] &* c.multipliers[0]
                    for p in 1 ..< ngram { mixed ^= key[p] &* c.multipliers[p] }
                    let low = (ngram - 2) * headsPerNGram
                    for head in low ..< low + headsPerNGram {
                        var r = mixed % c.sizes[head]
                        if r != 0, (r < 0) != (c.sizes[head] < 0) { r += c.sizes[head] }
                        gids.append(Int(r + c.offsets[head]))
                    }
                }
            }
            let rows = host.rows(
                globalIds: gids, shape: [1, missKeys.count, ngramHeads])
                .reshaped(1, missKeys.count, -1)
            for (i, slot) in missPositions.enumerated() {
                let row = rows[0..., i ..< (i + 1), 0...]
                parts[slot] = row
                if memo.count >= memoCap { memo.removeAll(keepingCapacity: true) }
                memo[keys[slot]] = row
            }
        }

        let assembled = parts.count == 1
            ? parts[0]!
            : concatenated(parts.map { $0! }, axis: 1)
        return assembled.asType(dtype)
    }

    static func invalidate() {
        memo.removeAll(keepingCapacity: false)
        constants.removeAll(keepingCapacity: false)
    }
}
