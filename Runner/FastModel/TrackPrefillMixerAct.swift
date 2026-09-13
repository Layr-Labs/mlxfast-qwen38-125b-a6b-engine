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
            grid: (5 * 32, ((rows + 31) / 32) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[1, rows, 320]], outputDTypes: [.bfloat16])[0]
    }

    /// MLXFAST-MIXACTBM: the down projection is 1024x320x10240, so the stock
    /// 64-row M tile leaves only 16 M tiles x 5 N tiles = 80 threadgroups for a
    /// 40-core GPU. A 32-row M tile doubles that to 160. The K traversal is
    /// untouched: `for (k = 0; k < K; k += BK)` with the same BK = 64 and the
    /// same `kk1` MMAs inside, so every output element still accumulates its K
    /// blocks in the same order -- only which rows share a threadgroup changes.
    /// `TM = SM / 16` becomes 1 instead of 2 and the register tile halves.
    static let source = #"""
        threadgroup T Ws[64 * 72];
        track_mixer_act_dense<T, 32, 4, true, 32, 64, 64, 2, 2>(
            w, scales, biases, x, y, Ws, 10240, 320, M,
            threadgroup_position_in_grid, thread_index_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
