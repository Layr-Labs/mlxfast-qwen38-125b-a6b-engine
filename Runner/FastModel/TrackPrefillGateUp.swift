import Foundation
import MLX

enum TrackPrefillGateUp {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_FUSED_GATE_UP"] != "0"

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
        name: "track_prefill_fused_gate_up",
        inputNames: ["x", "wg", "sg", "bg", "wu", "su", "bu", "indices"],
        outputNames: ["activated"],
        source: source,
        header: TrackFastKernels.exactHeader + metalHeader,
        ensureRowContiguous: true)

    static func apply(_ m: TrackMoE, x: MLXArray, indices: MLXArray) -> MLXArray? {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            m.expertBits == 4, m.expertGroupSize == 32,
            x.ndim == 3, x.dim(1) == 1, x.dim(2) == 2560, x.dtype == .bfloat16,
            x.dim(0) >= 2048, x.dim(0) < 512 * 64,
            indices.size == x.dim(0), indices.dtype == .uint32
        else { return nil }
        let g = m.expertGate
        let u = m.expertUp
        guard g.w.shape == [512, 640, 320], u.w.shape == g.w.shape,
            g.s.shape == [512, 640, 80], g.b.shape == g.s.shape,
            u.s.shape == g.s.shape, u.b.shape == g.s.shape,
            g.w.dtype == .uint32, u.w.dtype == .uint32,
            g.s.dtype == .bfloat16, g.b.dtype == .bfloat16,
            u.s.dtype == .bfloat16, u.b.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(0)
        return kernel(
            [x, g.w, g.s, g.b, u.w, u.s, u.b, indices],
            template: [("T", x.dtype), ("M", rows), ("N", 640), ("K", 2560)],
            grid: (10 * 32, ((rows + 31) / 32) * 2, 2),
            threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 640]], outputDTypes: [.bfloat16])[0]
    }

    static let source = #"""
        threadgroup T Wg[64 * 72];
        threadgroup T Wu[64 * 72];
        threadgroup T As[32 * 72];
        track_prefill_gate_up<T, 32, 4, 32, 64, 64, 2, 2, true>(
            x, wg, sg, bg, wu, su, bu, indices, activated,
            M, N, K, Wg, Wu, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
