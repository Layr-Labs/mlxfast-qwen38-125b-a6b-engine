import MLX
import MLXLLM

final class TrackRopeTables {
    private let capacity: Int
    private let rotaryDims: Int
    private let cos: MLXArray
    private let sin: MLXArray
    private let bf16Cos: MLXArray
    private let bf16Sin: MLXArray

    init?(rotary: Qwen4ExpRotary, rotaryDims: Int, indexerBudget: Int) {
        guard let capacity = TrackRopeCacheBounds.capacity(for: indexerBudget), rotaryDims > 0
        else { return nil }
        self.capacity = capacity
        self.rotaryDims = rotaryDims
        let (rawCos, rawSin) = rotary.cosSin(qwen4ExpPositions(offset: 0, count: capacity))
        let cos = rawCos.reshaped(capacity, rotaryDims)
        let sin = rawSin.reshaped(capacity, rotaryDims)
        let bf16Cos = cos.asType(.bfloat16)
        let bf16Sin = sin.asType(.bfloat16)
        eval(cos, sin, bf16Cos, bf16Sin)
        self.cos = cos
        self.sin = sin
        self.bf16Cos = bf16Cos
        self.bf16Sin = bf16Sin
    }

    func slice(offset: Int, count: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray)? {
        guard let range = TrackRopeCacheBounds.range(offset: offset, count: count, capacity: capacity)
        else { return nil }
        if dtype == .bfloat16 {
            return (
                bf16Cos[range].reshaped(count, rotaryDims),
                bf16Sin[range].reshaped(count, rotaryDims))
        }
        return (
            cos[range].asType(dtype).reshaped(count, rotaryDims),
            sin[range].asType(dtype).reshaped(count, rotaryDims))
    }
}
