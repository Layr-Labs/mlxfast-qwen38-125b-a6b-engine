import Foundation
import MLX
import MLXLLM

final class TrackRotaryTableCache: @unchecked Sendable {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_ROTARY_TABLE_CACHE"] != "0"
    private let lock = NSLock()
    private var tables: (cos: MLXArray, sin: MLXArray)?

    func get(
        rotary: Qwen4ExpRotary, offset: Int, count: Int, dtype: DType, capacity: Int
    ) -> (cos: MLXArray, sin: MLXArray)? {
        guard Self.enabled, StreamOrDevice.default.stream == Stream.gpu,
            dtype == .bfloat16, capacity == 2048, rotary.dimensions == 64,
            offset >= 0, count > 0, offset <= capacity, count <= capacity - offset
        else { return nil }
        lock.lock()
        defer { lock.unlock() }
        if tables == nil {
            let (c, s) = rotary.cosSin(qwen4ExpPositions(offset: 0, count: capacity))
            tables = (
                c.asType(dtype).reshaped(capacity, rotary.dimensions),
                s.asType(dtype).reshaped(capacity, rotary.dimensions))
        }
        let cached = tables!
        return (
            cached.cos[offset..<(offset + count), 0...],
            cached.sin[offset..<(offset + count), 0...])
    }
}
