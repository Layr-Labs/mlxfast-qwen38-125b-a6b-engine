import Foundation
import MLX
import MLXNN

enum TrackGDNProjectionPair {
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
        name: "track_gdn_projection_pair",
        inputNames: ["x", "w0", "s0", "b0", "w1", "s1", "b1"],
        outputNames: ["y0", "y1"], source: source,
        header: TrackPrefillIndirect.metalHeader + TrackFastKernels.exactHeader
            + TrackPrefillMixerAct.denseHeader,
        ensureRowContiguous: true)

    static func apply(_ parts: [TrackProj], x: MLXArray) -> [MLXArray]? {
        guard supportsNAX, StreamOrDevice.default.stream === Stream.gpu,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 64,
            x.dim(2) == 2560, x.dtype == .bfloat16, parts.count == 4,
            case .quant(let q) = parts[0], case .quant(let z) = parts[1],
            q.groupSize == 32, z.groupSize == 32, q.bits == 4, z.bits == 4,
            q.mode == .affine, z.mode == .affine,
            q.weight.shape == [10240, 320], z.weight.shape == [6144, 320],
            q.weight.dtype == .uint32, z.weight.dtype == .uint32,
            q.scales.shape == [10240, 80], z.scales.shape == [6144, 80],
            q.scales.dtype == .bfloat16, z.scales.dtype == .bfloat16,
            let qb = q.biases, let zb = z.biases,
            qb.shape == q.scales.shape, zb.shape == z.scales.shape,
            qb.dtype == .bfloat16, zb.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(1)
        let output = kernel(
            [x, q.weight, q.scales, qb, z.weight, z.scales, zb],
            template: [("T", x.dtype), ("M", rows)],
            grid: (256 * 32, ((rows + 63) / 64) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[1, rows, 10240], [1, rows, 6144]],
            outputDTypes: [.bfloat16, .bfloat16])
        return [output[0], output[1], parts[2].apply(x), parts[3].apply(x)]
    }

    static let source = #"""
        uint3 tile = threadgroup_position_in_grid;
        const bool first = tile.x < 160;
        if (!first) { tile.x -= 160; }
        alignas(16) threadgroup T Ws[64 * 72];
        track_mixer_act_dense<T, 32, 4, true, 64, 64, 64, 2, 2, false>(
            first ? w0 : w1, first ? s0 : s1, first ? b0 : b1,
            x, first ? y0 : y1, Ws, 2560, first ? 10240 : 6144, M,
            tile, thread_index_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
