// Decode lm_head GEMV. Same 4-bit qdot order as MLX qmv_fast. The K loop is the
// existing software-pipelined walk (next block's packs, scales, biases and
// activations are in registers before this block's products). The four reduced
// rows are written with one vec<T,4> store instead of four scalar scatters.
// Only the one-token head shape is served. Every other call stays on base.head.

import Foundation
import MLX
import MLXNN

enum TrackFastLMHead {
    static let source = """
        const int tile = int(threadgroup_position_in_grid.y);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int out_row = tile * 8 + sg * 4;
        const uint lid = thread_index_in_simdgroup;
        float result[4];
        qmv_fast_reg_pf2<T, 32, 4>(w, scales, biases, x, K, out_row, lid, result);
        if (lid == 0) {
            *reinterpret_cast<device vec<T, 4>*>(y + out_row) = vec<T, 4>(
                static_cast<T>(result[0]),
                static_cast<T>(result[1]),
                static_cast<T>(result[2]),
                static_cast<T>(result[3]));
        }
        """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_lm_head_qmv",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: source,
        header: TrackFastMoEKernels.helpersCore + TrackFastMoEKernels.pipelinedHelpers,
        ensureRowContiguous: true)

    /// Logits for a one-token hidden state. Nil means the caller must use the
    /// ordinary head (wrong shape, bias, quantization, or device).
    static func logits(_ hidden: MLXArray, model: Module) -> MLXArray? {
        guard StreamOrDevice.default.stream === Stream.gpu,
            let linear = model.children()[unwrapping: "lm_head"] as? QuantizedLinear,
            linear.bias == nil,
            linear.bits == 4,
            linear.groupSize == 32,
            linear.mode == .affine,
            let biases = linear.biases
        else { return nil }
        let k = hidden.dim(-1)
        let rows = linear.weight.dim(0)
        let lead = hidden.shape.dropLast().reduce(1, *)
        guard k == 2560, rows % 8 == 0, lead == 1, hidden.dtype == .bfloat16,
            linear.weight.dim(1) == k / 8, biases.dim(0) == rows
        else { return nil }
        let y = kernel(
            [linear.weight, linear.scales, biases, hidden.reshaped(k)],
            template: [("T", hidden.dtype), ("K", k)],
            grid: (32, rows / 4, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows]],
            outputDTypes: [hidden.dtype])[0]
        return y.reshaped(hidden.shape.dropLast() + [rows])
    }
}
