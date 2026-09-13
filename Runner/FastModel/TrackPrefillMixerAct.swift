import Foundation
import MLX
import MLXNN

enum TrackPrefillMixerAct {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_MIXER_SILU_EPILOGUE"] != "0"

    private static let supportsNAX: Bool = {
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return false
        }
        let arch = GPU.deviceInfo().architecture
        guard let generation = Int(arch.dropLast().suffix(2)), let family = arch.last else {
            return false
        }
        return generation >= (family == "p" ? 18 : 17)
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_mixer_silu_epilogue",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: source,
        header: TrackPrefillIndirect.metalHeader + TrackFastKernels.exactHeader + denseHeader,
        ensureRowContiguous: true)

    static func apply(_ projection: TrackProj, x: MLXArray, width: Int) -> MLXArray? {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            width == 320, x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 1024,
            x.dim(2) == 10240, x.dtype == .bfloat16,
            case .quant(let q) = projection, q.groupSize == 32, q.bits == 4,
            q.mode == .affine, q.weight.shape == [320, 1280], q.weight.dtype == .uint32,
            q.scales.shape == [320, 320], q.scales.dtype == .bfloat16,
            let biases = q.biases, biases.shape == q.scales.shape, biases.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(1)
        return kernel(
            [x, q.weight, q.scales, biases], template: [("T", x.dtype), ("M", rows)],
            grid: (5 * 32, ((rows + 63) / 64) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[1, rows, 320]], outputDTypes: [.bfloat16])[0]
    }

    static let source = #"""
        threadgroup T Ws[64 * 72];
        track_mixer_act_dense<T, 32, 4, true, 64, 64, 64, 2, 2>(
            w, scales, biases, x, y, Ws, 10240, 320, M,
            threadgroup_position_in_grid, thread_index_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
