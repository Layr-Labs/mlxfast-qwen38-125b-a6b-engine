import Foundation
import MLX

/// Final-query producer layout: Q/gate are projected for one row, K/V for
/// the complete prefill window. Every head's normalization body is inherited.
enum TrackTerminalAttention {
    private static let source = TrackFastKernels.attnPrepSource
        .replacingOccurrences(
            of: "if (isQ) { hh = h; src = row * QW + h * D; }",
            with: "if (isQ) { if (s != S - 1) return; hh = h; src = h * D; }")
        .replacingOccurrences(of: "src = row * QW + 2 * HQ * D + hh * D;", with: "src = row * HK * D + hh * D;")
        .replacingOccurrences(of: "src = row * QW + 2 * HQ * D + HK * D + hh * D;", with: "src = row * HK * D + hh * D;")
        .replacingOccurrences(of: "qkv[src + d]", with: "vproj[src + d]")
        .replacingOccurrences(of: "qkv[src + lid * N_READS + i]", with: "(isQ ? qkv : kproj)[src + lid * N_READS + i]")
        .replacingOccurrences(of: "qout + ((b * HQ + hh) * S + s) * D", with: "qout + hh * D")

    nonisolated(unsafe) private static let kernel = MLXFast.metalKernel(
        name: "track_terminal_qnorm_rope_full_kv_v1",
        inputNames: ["qkv", "kproj", "vproj", "qnorm", "knorm", "cosb", "sinb"],
        outputNames: ["qout", "kout", "vout"], source: source,
        header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func prepare(
        qGate: MLXArray, k: MLXArray, v: MLXArray, qNorm: MLXArray, kNorm: MLXArray,
        cos: MLXArray, sin: MLXArray, heads: Int, kvHeads: Int,
        headDim: Int, rotaryDims: Int, eps: Float
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let S = k.dim(1)
        let result = kernel(
            [qGate, k, v, qNorm, kNorm, cos, sin],
            template: [("InT", qGate.dtype), ("HQ", heads), ("HK", kvHeads),
                       ("D", headDim), ("ROT", rotaryDims), ("S", S),
                       ("QW", qGate.dim(2)), ("EPS_BITS", Int(eps.bitPattern))],
            grid: (headDim / 4, heads + 2 * kvHeads, S), threadGroup: (headDim / 4, 1, 1),
            outputShapes: [[1, heads, 1, headDim], [1, kvHeads, S, headDim], [1, kvHeads, S, headDim]],
            outputDTypes: [qGate.dtype, qGate.dtype, qGate.dtype])
        return (result[0], result[1], result[2])
    }
}
