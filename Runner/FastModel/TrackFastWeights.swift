// Quantized weight bundle (4-bit affine) with MLX's own matmul; shared by the
// model and the kernel unit tests.

import Foundation
import MLX
import MLXNN

struct TrackQuantWeight {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    var rows: Int { weight.dim(0) }

    func apply(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    func compatible(_ other: TrackQuantWeight) -> Bool {
        groupSize == other.groupSize && bits == other.bits && mode == other.mode
            && weight.dim(1) == other.weight.dim(1) && (biases == nil) == (other.biases == nil)
            && weight.dtype == other.weight.dtype
    }

    static func concat(_ parts: [TrackQuantWeight]) -> TrackQuantWeight {
        precondition(!parts.isEmpty)
        let first = parts[0]
        for p in parts.dropFirst() { precondition(first.compatible(p)) }
        let w = concatenated(parts.map(\.weight), axis: 0)
        let s = concatenated(parts.map(\.scales), axis: 0)
        let b = first.biases == nil ? nil : concatenated(parts.map { $0.biases! }, axis: 0)
        return TrackQuantWeight(
            weight: w, scales: s, biases: b, groupSize: first.groupSize, bits: first.bits,
            mode: first.mode)
    }

    func rowsReordered(_ order: [Int32]) -> TrackQuantWeight {
        let idx = MLXArray(order)
        return TrackQuantWeight(
            weight: weight[idx], scales: scales[idx], biases: biases.map { $0[idx] },
            groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Harness-only profiling: when `prefill` is set, wide windows eval after
/// every block and accumulate wall time per block kind.
public enum TrackFastProfile {
    nonisolated(unsafe) public static var prefill: [String: Double]? = nil
    /// Smallest window the ticks apply to (9 = prefill only; 2 = also MTP verify windows).
    nonisolated(unsafe) public static var minWindow: Int = 9
    nonisolated(unsafe) public static var windows: Int = 0
    /// Prefill-width knob exposed for the bench drivers (kernel selection only; results are exact either way).
    nonisolated(unsafe) public static var wideNormMinS: Int {
        get { TrackFastKernels.wideNormMinS }
        set { TrackFastKernels.wideNormMinS = newValue }
    }
    static func tick(_ key: String, _ t0: inout Double, _ arrays: [MLXArray]) {
        guard prefill != nil else { return }
        eval(arrays)
        let t = CFAbsoluteTimeGetCurrent()
        prefill![key, default: 0] += t - t0
        t0 = t
    }
}
