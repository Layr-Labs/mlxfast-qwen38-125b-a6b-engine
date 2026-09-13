import Cmlx
import Foundation
import MLX
import MLXLLM

final class TrackPLELookup: @unchecked Sendable {
    let history: [Int64]
    private let source: Qwen4ExpNGramTable
    private let ids: [Int]
    private let shape: [Int]
    private let condition = NSCondition()
    private var result: MLXArray?

    static func isAvailable(_ array: MLXArray) -> Bool {
        var available = false
        return _mlx_array_is_available(&available, array.ctx) == 0 && available
    }

    init(source: Qwen4ExpNGramTable, ids: [Int], shape: [Int], history: [Int64]) {
        self.source = source
        self.ids = ids
        self.shape = shape
        self.history = history
        DispatchQueue.global(qos: .userInitiated).async {
            let rows = self.source.rows(globalIds: self.ids, shape: self.shape)
            self.condition.lock()
            self.result = rows
            self.condition.signal()
            self.condition.unlock()
        }
    }

    func wait() -> MLXArray {
        condition.lock()
        defer { condition.unlock() }
        while result == nil { condition.wait() }
        return result!
    }
}
