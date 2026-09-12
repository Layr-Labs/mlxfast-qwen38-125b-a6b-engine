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

    /// Row count past which the repository's own wide replica is used instead
    /// of MLX's op at decode widths.
    ///
    /// `TrackFastMoEKernels.wideCheck` is `qmv_wide_impl` reproduced verbatim
    /// -- same lane-to-group assignment, same decode, same element order, same
    /// shuffle ladder -- so it returns the same bits as the op it stands in
    /// for, and it is one launch either way. It carries four buffers where the
    /// op also carries its shape and stride tables. At narrow row counts the
    /// two measure the same, so the threshold keeps those on the op.
    static let wideReplicaMinRows = 8192

    func apply(_ x: MLXArray) -> MLXArray {
        if bits == 4, mode == .affine, let b = biases, x.ndim == 3, x.dim(0) == 1,
            x.dim(1) >= 2, x.dim(1) <= 8, weight.dim(0) >= Self.wideReplicaMinRows,
            weight.dim(0) % 8 == 0
        {
            let S = x.dim(1), K = x.dim(2)
            return TrackFastMoEKernels.wideCheck(
                w: weight, scales: scales, biases: b, x: x.reshaped(S, K),
                groupSize: groupSize, bits: bits
            ).reshaped(1, S, weight.dim(0))
        }
        return quantizedMM(
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
