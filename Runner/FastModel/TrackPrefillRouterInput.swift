import Foundation
import MLX

enum TrackPrefillRouterInput {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_ROUTER_INPUT_EMISSION"] != "0"

    static let source: String = {
        let anchor = "input[row * H + d] = acc;"
        let original = TrackFastKernels.mixSource
        precondition(original.components(separatedBy: anchor).count == 2)
        return original.replacingOccurrences(
            of: anchor,
            with: anchor + "\ninputF32[row * H + d] = static_cast<float>(acc);")
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_hc_mix_router_input",
        inputNames: ["w", "normed", "inj"], outputNames: ["input", "inject", "inputF32"],
        source: source, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func apply(
        w: MLXArray, normed: MLXArray, inj: MLXArray,
        hcCount: Int, hidden: Int, hasInject: Bool
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray)? {
        guard enabled, StreamOrDevice.default.stream == Stream.gpu,
            hcCount == 4, hidden == 2560, hasInject,
            w.ndim == 3, w.dim(0) == 1, w.dim(1) > 8,
            w.dim(2) == hcCount * hidden, w.dtype == .bfloat16,
            normed.shape == w.shape, normed.dtype == w.dtype,
            inj.shape == [1, w.dim(1), hcCount], inj.dtype == w.dtype
        else { return nil }
        let rows = w.dim(1)
        let outs = kernel(
            [w, normed, inj],
            template: [
                ("InT", w.dtype), ("H", hidden), ("W", hcCount * hidden), ("HC", hcCount),
                ("LW", hcCount), ("HAS_INJECT", true),
            ],
            grid: (hidden, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, hidden], [1, rows, hcCount], [1, rows, hidden]],
            outputDTypes: [.bfloat16, .bfloat16, .float32])
        return (outs[0], outs[1], outs[2])
    }
}
