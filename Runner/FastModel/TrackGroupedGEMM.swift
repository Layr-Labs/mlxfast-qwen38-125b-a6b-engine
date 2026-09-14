// TrackGroupedGEMM.swift -- prefill-only sorted-pair grouped expert GEMM.
//
// Sort (token, expert) pairs by expert (original slot breaks ties), gather
// each token activation once into that expert's contiguous block, run the
// same `gather_qmm` projections SwitchLinear already uses, and combine
// through the inverse permutation. Intra-row reduction stays the gather_qmm
// kernel (rhs grouped GEMM when B/E >= 4, else gather_qmv): only the gather
// and launch layout change.
//
// Decode windows (S <= 8) never enter this path. `TRACK_GROUPED_GEMM=0`
// restores the previous prefill gather.

import Foundation
import MLX
import MLXLMCommon

enum TrackGroupedGEMM {
    /// Default ON. Set `TRACK_GROUPED_GEMM=0` to fall back.
    static let enabled =
        ProcessInfo.processInfo.environment["TRACK_GROUPED_GEMM"] != "0"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var slotTables: [Int: MLXArray] = [:]

    /// Cached `0 ..< slots` as int64, one array per window size.
    private static func slotIndex(_ slots: Int) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let t = slotTables[slots] { return t }
        let t = MLXArray.arange(slots, dtype: .int64)
        eval(t)
        slotTables[slots] = t
        return t
    }

    /// Expert-major order; the original slot index breaks ties.
    static func sortPairs(_ flatIDs: MLXArray) -> (
        order: MLXArray, inverse: MLXArray, sortedIDs: MLXArray
    ) {
        let slots = flatIDs.size
        let keys = flatIDs.asType(.int64) * slots + slotIndex(slots)
        let order = argSort(keys)
        return (order, argSort(order), flatIDs[order])
    }

    static func apply(
        _ m: TrackMoE, _ x: MLXArray, indices: MLXArray, weights: MLXArray,
        shared: MLXArray, gate: MLXArray
    ) -> MLXArray? {
        guard enabled, TrackP12Prefill.eligible(x), indices.size >= 64,
            weights.dtype == .float32, StreamOrDevice.default.stream === Stream.gpu,
            let parts = m.p12SortedParts, !m.switchMLP.hasFusedGateUp
        else { return nil }

        let B = x.dim(0), S = x.dim(1), H = x.dim(2), K = indices.dim(-1)
        let slots = indices.size
        let sorted = sortPairs(indices.flattened())
        let tokenRows = sorted.order.floorDivide(K)
        let gathered = x.reshaped(S, H)[tokenRows].reshaped(slots, 1, H)
        let up = parts.up(gathered, sorted.sortedIDs, sortedIndices: true)
        let gateAct = parts.gate(gathered, sorted.sortedIDs, sortedIndices: true)
        let activated = compiledSiluProduct(gateAct, up)
        let down = parts.down(activated, sorted.sortedIDs, sortedIndices: true)
        return TrackP12Prefill.combineSorted(
            down: down, weights: weights, shared: shared, gate: gate,
            inverse: sorted.inverse, B: B, S: S, H: H, K: K)
    }
}
