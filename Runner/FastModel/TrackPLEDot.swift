import Foundation
import MLX

enum TrackPLEDot {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PLE_FUSED_DOT"] != "0"

    static func makeSource(_ original: String) -> String {
        func replace(_ source: String, _ old: String, _ new: String) -> String {
            precondition(source.components(separatedBy: old).count == 2)
            return source.replacingOccurrences(of: old, with: new)
        }
        var source = replace(
            original, "threadgroup float ksums[32];",
            "threadgroup InT dot_sums[20];\nthreadgroup float ksums[32];")
        source = replace(
            source, "const float qinv = metal::precise::rsqrt(qacc / (float)H + eps);",
            """
            const float qinv = metal::precise::rsqrt(qacc / (float)H + eps);
            InT total = InT(0);
            """)
        source = replace(
            source, "prod[base + d] = kn * qn;",
            "const InT product = kn * qn;\ntotal = product + total;")
        return source + """

            total = simd_sum(total);
            if (lane == 0) { dot_sums[sg] = total; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = lid < 20 ? dot_sums[lid] : InT(0);
            total = simd_sum(total);
            if (lid == 0) { dot[row * HC + hc] = total; }
            """
    }

    private static let kernel = MLXFast.metalKernel(
        name: "track_ple_fused_dot",
        inputNames: ["keyFlat", "stream", "kscale", "qscale", "eps"],
        outputNames: ["dot"], source: makeSource(TrackFastPLEKernels.prodSource),
        header: TrackFastPLEKernels.header, ensureRowContiguous: true)

    static func apply(
        keyFlat: MLXArray, stream: MLXArray, kScale: MLXArray, qScale: MLXArray,
        hcCount: Int, hidden: Int, eps: Float
    ) -> MLXArray? {
        guard enabled, StreamOrDevice.default.stream == Stream.gpu,
            hidden == 2560, hcCount == 4, keyFlat.ndim == 3,
            keyFlat.dim(0) == 1, keyFlat.dim(1) >= 8, keyFlat.dim(2) == 10240,
            keyFlat.dtype == .bfloat16, stream.shape == keyFlat.shape,
            stream.dtype == keyFlat.dtype, kScale.shape == [10240],
            qScale.shape == kScale.shape, kScale.dtype == keyFlat.dtype,
            qScale.dtype == keyFlat.dtype
        else { return nil }
        let steps = keyFlat.dim(1)
        return kernel(
            [keyFlat, stream, kScale, qScale, MLXArray(eps)],
            template: [("InT", keyFlat.dtype), ("H", hidden), ("W", 10240), ("HC", hcCount)],
            grid: (640, 4, steps), threadGroup: (640, 1, 1),
            outputShapes: [[1, steps, 4, 1]], outputDTypes: [keyFlat.dtype])[0]
    }
}
