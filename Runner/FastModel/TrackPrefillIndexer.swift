import Foundation
import MLX
import MLXNN

enum TrackPrefillIndexer {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_INDEXER_LIVE_COLUMNS"] != "0"

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
        name: "track_prefill_indexer_live_columns",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: source, header: TrackPrefillIndirect.metalHeader + denseHeader,
        ensureRowContiguous: true)

    static func apply(_ projection: TrackProj, x: MLXArray) -> MLXArray? {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 1024,
            x.dim(2) == 2560, x.dtype == .bfloat16,
            case .quant(let q) = projection, q.groupSize == 32, q.bits == 4,
            q.mode == .affine, let biases = q.biases,
            q.weight.shape == [640, 320], q.weight.dtype == .uint32,
            q.scales.shape == [640, 80], biases.shape == q.scales.shape,
            q.scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(1)
        return kernel(
            [x, q.weight, q.scales, biases], template: [("T", x.dtype), ("M", rows)],
            grid: (2 * 32, ((rows + 63) / 64) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[1, rows, 128]], outputDTypes: [.bfloat16])[0]
    }

    static let source = #"""
        threadgroup T Ws[64 * 72];
        track_indexer_dense<T, 32, 4, true, 64, 64, 64, 2, 2>(
            w + 512 * 320, scales + 512 * 80, biases + 512 * 80,
            x, y, Ws, 2560, 128, M, threadgroup_position_in_grid,
            thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
        """#
}
