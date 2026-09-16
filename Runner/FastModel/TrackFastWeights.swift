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
    /// Peak MLX active memory (bytes) reached inside each block kind, when set.
    nonisolated(unsafe) public static var memory: [String: Int]? = nil
    /// `TRACK_FAST_PROFILE=1` turns the ticks on in the worker and prints one
    /// report per wide window to stderr (diagnostic; ticks eval per block).
    public static let environmentEnabled: Bool =
        ProcessInfo.processInfo.environment["TRACK_FAST_PROFILE"] == "1"
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
        if memory != nil {
            memory![key] = max(memory![key] ?? 0, Memory.peakMemory)
            Memory.peakMemory = 0  // resets the peak to the current active level
        }
    }

    /// One stderr line per wide window: wall per block kind, peak active
    /// memory per block kind, and the allocator's state now.
    static func report(window: Int, offset: Int) {
        guard let times = prefill else { return }
        let gb = { (b: Int) in String(format: "%.1f", Double(b) / 1e9) }
        let keys = times.keys.sorted()
        let parts = keys.map { k in
            "\(k)=\(String(format: "%.2f", times[k]!))s/\(gb(memory?[k] ?? 0))GB"
        }
        FileHandle.standardError.write(
            Data(("track-fast-profile: window S=\(window) offset=\(offset) "
                + "active=\(gb(Memory.activeMemory))GB cache=\(gb(Memory.cacheMemory))GB "
                + parts.joined(separator: " ") + "\n").utf8))
        prefill = [:]
        memory = [:]
    }
}
