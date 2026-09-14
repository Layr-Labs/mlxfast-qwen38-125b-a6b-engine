// TrackPLESSMStaging.swift -- persistent int32 buffers for the PLE n-gram SSM.
//
// Decode and capture each keep two already-eval'd arrays and alternate, so a
// step still in flight is not overwritten by the next step's host write.
// Values are stored through the eval'd host pointer; the returned MLXArray
// object is the same one every other round.

import Foundation
import MLX

final class TrackPLESSMStaging: @unchecked Sendable {
    static let hostPathMaxRows = 8

    let contextLength: Int
    let maxRows: Int

    private let lock = NSLock()
    private let decode: [MLXArray]
    private let decodePtr: [UnsafeMutablePointer<Int32>]
    private var decodeFlip = 0
    private let capture: [MLXArray]
    private let capturePtr: [UnsafeMutablePointer<Int32>]
    private let captureViews: [[MLXArray]]
    private var captureFlip = 0

    init(contextLength: Int, maxRows: Int = TrackPLESSMStaging.hostPathMaxRows) {
        precondition(contextLength >= 1, "TrackPLESSMStaging: contextLength must be >= 1")
        precondition(maxRows >= 1, "TrackPLESSMStaging: maxRows must be >= 1")
        self.contextLength = contextLength
        self.maxRows = maxRows

        func make(_ rows: Int) -> (MLXArray, UnsafeMutablePointer<Int32>) {
            let count = rows * contextLength
            let array = MLXArray([Int32](repeating: 0, count: count), [rows, contextLength])
            eval(array)
            return (array, Self.int32Pointer(array))
        }

        let d0 = make(1)
        let d1 = make(1)
        decode = [d0.0, d1.0]
        decodePtr = [d0.1, d1.1]

        let c0 = make(maxRows)
        let c1 = make(maxRows)
        capture = [c0.0, c1.0]
        capturePtr = [c0.1, c1.1]

        func views(_ storage: MLXArray) -> [MLXArray] {
            (1 ... maxRows).map { s in
                if s == maxRows { return storage }
                return asStrided(
                    storage, [s, contextLength], strides: [contextLength, 1], offset: 0)
            }
        }
        captureViews = [views(c0.0), views(c1.0)]
    }

    func writeDecode<C: Collection>(_ values: C) -> MLXArray where C.Element: BinaryInteger {
        precondition(
            values.count == contextLength,
            "TrackPLESSMStaging: decode write needs \(contextLength) ids, got \(values.count)")
        lock.lock()
        defer { lock.unlock() }
        let i = decodeFlip
        decodeFlip ^= 1
        let dest = decodePtr[i]
        var j = 0
        for value in values {
            dest[j] = Int32(value)
            j += 1
        }
        return decode[i]
    }

    func writeCapture(history: [Int64], newCount: Int) -> MLXArray {
        precondition(
            newCount >= 1 && newCount <= maxRows,
            "TrackPLESSMStaging: capture rows \(newCount) outside 1...\(maxRows)")
        precondition(
            history.count >= newCount + contextLength,
            "TrackPLESSMStaging: history length \(history.count) is short for "
                + "newCount \(newCount) context \(contextLength)")
        lock.lock()
        defer { lock.unlock() }
        let i = captureFlip
        captureFlip ^= 1
        let dest = capturePtr[i]
        for s in 0 ..< newCount {
            let base = s * contextLength
            let src = s + 1
            for k in 0 ..< contextLength {
                dest[base + k] = Int32(history[src + k])
            }
        }
        return captureViews[i][newCount - 1]
    }

    private static func int32Pointer(_ array: MLXArray) -> UnsafeMutablePointer<Int32> {
        let wrapped = array.asData(access: .noCopy)
        precondition(wrapped.dType == .int32, "TrackPLESSMStaging: staging is int32")
        precondition(
            wrapped.data.count == array.size * MemoryLayout<Int32>.size,
            "TrackPLESSMStaging: backing size \(wrapped.data.count) != \(array.size) Int32s")
        return wrapped.data.withUnsafeBytes { raw in
            UnsafeMutablePointer(mutating: raw.bindMemory(to: Int32.self).baseAddress!)
        }
    }
}
