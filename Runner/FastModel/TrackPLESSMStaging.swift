// TrackPLESSMStaging.swift -- persistent int32 buffers for the PLE n-gram SSM.
//
// Decode and capture each keep two already-eval'd arrays and alternate, so a
// step still in flight is not overwritten by the next step's host write.
// Decode and capture writes use a noCopy mapping taken for the duration of
// the store only. A pointer cached from init or escaped from withUnsafeBytes
// is not the live GPU buffer (SIGSEGV after later evals). Same MLXArray
// object every other round.

import Foundation
import MLX

final class TrackPLESSMStaging: @unchecked Sendable {
    static let hostPathMaxRows = 8

    let contextLength: Int
    let maxRows: Int

    private let lock = NSLock()
    private let decode: [MLXArray]
    private let decodeView: [MLXArray]
    private var decodeFlip = 0
    private let capture: [MLXArray]
    private let captureViews: [[MLXArray]]
    private var captureFlip = 0

    init(contextLength: Int, maxRows: Int = TrackPLESSMStaging.hostPathMaxRows) {
        precondition(contextLength >= 1, "TrackPLESSMStaging: contextLength must be >= 1")
        precondition(maxRows >= 1, "TrackPLESSMStaging: maxRows must be >= 1")
        self.contextLength = contextLength
        self.maxRows = maxRows

        func make(_ rows: Int) -> MLXArray {
            let count = rows * contextLength
            let array = MLXArray([Int32](repeating: 0, count: count), [rows, contextLength])
            eval(array)
            return array
        }

        // 1-row noCopy maps are not the live asArray/GPU buffer (writes
        // read back as zeros). Decode uses the same multi-row storage as
        // capture and returns a cached 1-row view of row 0.
        let d0 = make(maxRows)
        let d1 = make(maxRows)
        decode = [d0, d1]
        decodeView = [
            asStrided(d0, [1, contextLength], strides: [contextLength, 1], offset: 0),
            asStrided(d1, [1, contextLength], strides: [contextLength, 1], offset: 0),
        ]
        let c0 = make(maxRows)
        let c1 = make(maxRows)
        capture = [c0, c1]

        func views(_ storage: MLXArray) -> [MLXArray] {
            (1 ... maxRows).map { s in
                if s == maxRows { return storage }
                return asStrided(
                    storage, [s, contextLength], strides: [contextLength, 1], offset: 0)
            }
        }
        captureViews = [views(c0), views(c1)]
    }

    func writeDecode<C: Collection>(_ values: C) -> MLXArray where C.Element: BinaryInteger {
        precondition(
            values.count == contextLength,
            "TrackPLESSMStaging: decode write needs \(contextLength) ids, got \(values.count)")
        lock.lock()
        defer { lock.unlock() }
        let i = decodeFlip
        decodeFlip ^= 1
        eval(decode[i])
        let wrapped = decode[i].asData(access: .noCopy)
        precondition(wrapped.dType == .int32, "TrackPLESSMStaging: staging is int32")
        wrapped.data.withUnsafeBytes { raw in
            let dest = UnsafeMutablePointer(
                mutating: raw.bindMemory(to: Int32.self).baseAddress!)
            var j = 0
            for value in values {
                dest[j] = Int32(value)
                j += 1
            }
        }
        return decodeView[i]
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
        eval(capture[i])
        let wrapped = capture[i].asData(access: .noCopy)
        precondition(wrapped.dType == .int32, "TrackPLESSMStaging: staging is int32")
        wrapped.data.withUnsafeBytes { raw in
            let dest = UnsafeMutablePointer(
                mutating: raw.bindMemory(to: Int32.self).baseAddress!)
            for s in 0 ..< newCount {
                let base = s * contextLength
                let src = s + 1
                for k in 0 ..< contextLength {
                    dest[base + k] = Int32(history[src + k])
                }
            }
        }
        return captureViews[i][newCount - 1]
    }
}
