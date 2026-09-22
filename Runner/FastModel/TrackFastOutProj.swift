// One-token block output projection. Same 4-bit qdot order as MLX qmv_fast.
// The K loop is the existing software-pipelined walk. The four reduced rows are
// written with one vec<T, 4> store. Only N=2560, K=6144 is served. Every other
// projection stays on TrackProj.apply.

import Foundation
import MLX

enum TrackFastOutProj {
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
        name: "track_out_proj_qmv",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: source,
        header: TrackFastMoEKernels.helpersCore + TrackFastMoEKernels.pipelinedHelpers,
        ensureRowContiguous: true)

    /// One-token output projection. Nil means the caller must use `proj.apply`.
    static func y(_ x: MLXArray, proj: TrackProj) -> MLXArray? {
        guard StreamOrDevice.default.stream === Stream.gpu,
            case .quant(let q) = proj,
            q.bits == 4,
            q.groupSize == 32,
            q.mode == .affine,
            let biases = q.biases
        else { return nil }
        let k = x.dim(-1)
        let rows = q.rows
        let lead = x.shape.dropLast().reduce(1, *)
        guard k == 6144, rows == 2560, lead == 1, x.dtype == .bfloat16,
            q.weight.dim(1) == k / 8, biases.dim(0) == rows
        else { return nil }
        let y = kernel(
            [q.weight, q.scales, biases, x.reshaped(k)],
            template: [("T", x.dtype), ("K", k)],
            grid: (32, rows / 4, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows]],
            outputDTypes: [x.dtype])[0]
        return y.reshaped(x.shape.dropLast() + [rows])
    }
}
