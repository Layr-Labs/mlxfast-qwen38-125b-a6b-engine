// TrackFastConstants.swift -- cached scalar and index tensors for the fast path.
//
// The forward pass hands several small tensors to custom kernels as *inputs*:
// the K and N widths, the recurrence length T, and the `xrow` map that says
// which row of `x` each (row, expert) pair reads. Every one of them is a pure
// function of a shape that is fixed for the life of the process -- a window
// size, an expert count, a projection width -- and the decode hot path rebuilds
// them on every call: roughly 276 constructions per forward across the 48
// layers and their MoE blocks.
//
// Each construction allocates a host buffer, wraps it in an mlx_array and
// materializes it on the device. This caches them instead, keyed by value.
//
// THIS CHANGES NO ARITHMETIC. The same values reach the same kernel slots; the
// only difference is that an identical constant is built once rather than
// hundreds of times per step. Nothing here is prompt-dependent or window-
// dependent beyond the shape key, and no accumulation order is touched.

import Foundation
import MLX

enum TrackFastConstants {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var scalarCache: [Int32: MLXArray] = [:]
    nonisolated(unsafe) private static var xrowCache: [Int: MLXArray] = [:]

    /// A cached 0-dim int32 tensor, for kernel inputs that are static widths.
    static func i32(_ value: Int) -> MLXArray {
        let key = Int32(value)
        lock.lock()
        defer { lock.unlock() }
        if let hit = scalarCache[key] { return hit }
        let made = MLXArray(key)
        scalarCache[key] = made
        return made
    }

    /// `xrow[b * K + k] = b`: the row of `x` that the (row, expert) pair at
    /// flat index `b * K + k` reads. Depends only on window size and topK.
    static func xrow(S: Int, K: Int) -> MLXArray {
        let key = S &* 4096 &+ K
        lock.lock()
        defer { lock.unlock() }
        if let hit = xrowCache[key] { return hit }
        let made = MLXArray((0 ..< (S * K)).map { UInt32($0 / K) })
        xrowCache[key] = made
        return made
    }
}
