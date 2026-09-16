import Foundation
import MLX
import MLXLLM

/// Input-independent rotary values produced by the pinned implementation.
/// The cache is indexed only by position; it holds no token or model outputs.
/// Adapted from the historical Laguna position-atlas mechanism.
final class TrackRotaryAtlas {
    let cos: MLXArray
    let sin: MLXArray
    let length: Int
    let dimensions: Int

    init?(rotary: Qwen4ExpRotary, positionLimit: Int) {
        guard ProcessInfo.processInfo.environment["MLX_TRACK_ROTARY_ATLAS"] != "0",
            StreamOrDevice.default.stream === Stream.gpu,
            rotary.dimensions > 0, positionLimit > 0
        else { return nil }
        // Bound startup allocation independently of arbitrary configurations.
        let n = min(positionLimit, 4096)
        let d = rotary.dimensions
        let values = rotary.cosSin(qwen4ExpPositions(offset: 0, count: n))
        let c = values.0.asType(.bfloat16).reshaped(n, d)
        let s = values.1.asType(.bfloat16).reshaped(n, d)
        length = n
        dimensions = d
        cos = c
        sin = s
        eval(c, s)
    }

    func tables(offset: Int, count: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray)? {
        guard dtype == .bfloat16, count > 0, offset >= 0,
            offset <= length, count <= length - offset,
            StreamOrDevice.default.stream === Stream.gpu
        else { return nil }
        return (cos[offset ..< (offset + count)], sin[offset ..< (offset + count)])
    }
}
