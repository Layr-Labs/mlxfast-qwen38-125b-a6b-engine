import Foundation
import MLX

enum TrackStreamMaterialize {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PLE_STREAM_ONLY"] != "0"

    static let source = """
        const uint d = thread_position_in_grid.x;
        const uint hc = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        if (d >= H) { return; }
        const uint base = row * W + hc * H;
        const uint src = TILE ? row * H + d : base + d;
        InT r = residual[src];
        if (HAS_INJECT) {
            const InT inj_t = inject[row * HC + hc];
            const InT sp = out[row * H + d] * inj_t;
            r = r + sp;
        }
        stream[base + d] = r;
        """

    private static let kernel = MLXFast.metalKernel(
        name: "track_ple_stream_only", inputNames: ["residual", "out", "inject"],
        outputNames: ["stream"], source: source, ensureRowContiguous: true)

    static func apply(
        residual: MLXArray, out: MLXArray?, inject: MLXArray?, tile: Bool,
        hidden: Int, hcCount: Int
    ) -> MLXArray? {
        guard enabled, StreamOrDevice.default.stream == Stream.gpu,
            (out == nil) == (inject == nil),
            hidden == 2560, hcCount == 4, residual.ndim == 3,
            residual.dim(0) == 1, residual.dtype == .bfloat16,
            residual.dim(2) == (tile ? hidden : hidden * hcCount)
        else { return nil }
        let rows = residual.dim(1)
        let hasInject = out != nil && inject != nil
        if let out, let inject {
            guard out.shape == [1, rows, hidden], inject.shape == [1, rows, hcCount],
                out.dtype == residual.dtype, inject.dtype == residual.dtype
            else { return nil }
        }
        if !tile && !hasInject { return residual }
        return kernel(
            [residual, out ?? residual, inject ?? residual],
            template: [
                ("InT", residual.dtype), ("H", hidden), ("W", hidden * hcCount),
                ("HC", hcCount), ("TILE", tile), ("HAS_INJECT", hasInject),
            ],
            grid: (hidden, hcCount, rows), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, hidden * hcCount]], outputDTypes: [residual.dtype])[0]
    }
}
