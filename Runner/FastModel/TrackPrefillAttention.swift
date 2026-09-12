import Foundation
import MLX
import MLXNN

enum TrackPrefillAttention {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_FUSED_ATTENTION_PROJECTIONS"] != "0"

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

    static func prepare(_ projections: TrackMultiProj) -> TrackProj? {
        guard enabled, projections.parts.count == 4,
            projections.parts.map(\.rows) == [12288, 512, 512, 128],
            case .quant(let q)? = projections.fused,
            q.mode == .affine, q.groupSize == 32, q.bits == 4,
            q.weight.shape == [13440, 320], q.weight.dtype == .uint32,
            q.scales.shape == [13440, 80], q.scales.dtype == .bfloat16,
            let biases = q.biases, biases.shape == q.scales.shape,
            biases.dtype == .bfloat16
        else { return nil }
        return .quant(TrackQuantWeight(
            weight: q.weight[0..<13312], scales: q.scales[0..<13312],
            biases: biases[0..<13312], groupSize: q.groupSize, bits: q.bits, mode: q.mode))
    }

    static func apply(_ projection: TrackProj?, x: MLXArray) -> MLXArray? {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 1024,
            x.dim(2) == 2560, x.dtype == .bfloat16, let projection
        else { return nil }
        return projection.apply(x)
    }
}
