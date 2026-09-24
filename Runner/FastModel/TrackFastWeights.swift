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
    /// Load-time pair index of the metadata (see `TrackAffineLUT`); nil when
    /// the tensor does not qualify or the index is switched off.
    let meta: TrackAffineLUT?

    init(weight: MLXArray, scales: MLXArray, biases: MLXArray?, groupSize: Int, bits: Int, mode: QuantizationMode, meta: TrackAffineLUT? = nil) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.meta = meta
    }

    /// The same bundle with its metadata pair index built (load time only:
    /// the build reads the metadata back to the host).
    func withPairIndex() -> TrackQuantWeight {
        guard meta == nil, bits == 4, groupSize == 32, mode == .affine,
            let built = TrackAffineLUT.build(scales: scales, biases: biases)
        else { return self }
        return TrackQuantWeight(weight: weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: mode, meta: built)
    }

    var rows: Int { weight.dim(0) }

    func apply(_ x: MLXArray) -> MLXArray {
        // One-row windows: the reference's own qmv_fast / qmv walk, metadata
        // read through the pair index (same operands, same order, fewer bytes).
        if let meta, let y = TrackLUTGemv.apply(x, weight: weight, meta: meta, groupSize: groupSize, bits: bits) {
            return y
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
        let fused = TrackQuantWeight(
            weight: w, scales: s, biases: b, groupSize: first.groupSize, bits: first.bits,
            mode: first.mode)
        return parts.contains { $0.meta != nil } ? fused.withPairIndex() : fused
    }

    func rowsReordered(_ order: [Int32]) -> TrackQuantWeight {
        let idx = MLXArray(order)
        let reordered = TrackQuantWeight(
            weight: weight[idx], scales: scales[idx], biases: biases.map { $0[idx] },
            groupSize: groupSize, bits: bits, mode: mode)
        return meta != nil ? reordered.withPairIndex() : reordered
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
