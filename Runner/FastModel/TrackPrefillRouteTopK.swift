// TrackPrefillRouteTopK.swift -- the dedicated no-gate prefill route kernel:
// top-k + softmax over the router's split-K partials, without materializing
// the gate logit matrix.
//
// WHAT THE PARENT DOES, per MoE layer per prefill window
// (TrackPrefillRouter.apply -> TrackFastMoEKernels.route):
//
//   track_router_bf16_storage   writes partials [2, S, 512] f32
//   track_router_split_sum      reads both partitions, writes logits [S, 512]
//   track_moe_route             reads logits, walks top-k, writes idx + w
//
// The middle launch exists only to fold the two K partitions together. The
// route walk reads each logit exactly once, so the fold can live inside the
// walk's own load: `v = p0[e] + p1[e]` is the same two-operand float32 add
// `track_router_split_sum` performs, in the same order, on the same values.
// One launch and the whole [S, 512] logits write+read per layer go away.
//
// EXACTNESS. The kernel below is `TrackFastMoEKernels.routeSource` with ONE
// substituted load expression; every other line -- the lane-strided expert
// ownership, the k-step simd_max/simd_min walk, the lowest-index tie rule,
// the register-resident results, and the softmax_single_row epilogue -- is
// the decode path's source verbatim, so idx and w are bit-identical to the
// sum-then-route chain. The shared-expert gate stays out of this kernel
// (HAS_GATE = false): prefill computes it as a full GEMM elsewhere, which is
// why this is the no-gate variant.

import Foundation
import MLX

enum TrackPrefillRouteTopK {
    /// `TRACK_PREFILL_ROUTE_PARTIALS=0` restores the sum-then-route chain.
    static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_ROUTE_PARTIALS"] != "0"

    /// routeSource with the logit load reading both split-K partitions.
    /// `PM` is the row count, so partition 1 sits `PM * E` elements ahead.
    private static let partialsSource: String = {
        let anchor = "v[j] = (e < E) ? lr[e] : -INFINITY;"
        let fused =
            "v[j] = (e < E) ? (lr[e] + lr[(size_t)PM * (size_t)E + e]) : -INFINITY;"
        let source = TrackFastMoEKernels.routeSource
        precondition(
            source.components(separatedBy: anchor).count == 2,
            "TrackPrefillRouteTopK: route load anchor is no longer unique; re-audit")
        return source.replacingOccurrences(of: anchor, with: fused)
    }()

    nonisolated(unsafe) private static let kernel = MLXFast.metalKernel(
        name: "track_moe_route_partials",
        inputNames: ["logits", "x", "wg", "sgw", "bgw"],
        outputNames: ["idx", "w", "gate"],
        source: partialsSource,
        header: TrackFastKernels.mixerHeadHeader + TrackFastMoEKernels.wideHelpers,
        ensureRowContiguous: true)

    /// `(idx uint32 [1, M, K], w f32 [1, M, K])` straight from the router's
    /// `[2, M, E]` f32 partials, or nil outside the supported window.
    static func route(partials: MLXArray, x: MLXArray, topK: Int)
        -> (idx: MLXArray, w: MLXArray)?
    {
        guard enabled, partials.ndim == 3, partials.dim(0) == 2,
            partials.dtype == .float32, x.dtype == .bfloat16,
            x.dim(-1) % 256 == 0, topK > 0, topK <= 32
        else { return nil }
        let M = partials.dim(1), E = partials.dim(2)
        guard M >= 1, topK <= E else { return nil }
        let outs = kernel(
            // logits = partials; x/wg/sgw/bgw are dummies under HAS_GATE = false.
            [partials, x, x, x, x],
            template: [
                ("E", E), ("K", topK), ("T", x.dtype), ("GS", 32), ("BITS", 4),
                ("KD", x.dim(-1)), ("VPT", M), ("HAS_GATE", false), ("PM", M),
            ],
            grid: (32, M, 1), threadGroup: (32, 1, 1),
            outputShapes: [[M, topK], [M, topK], [M]],
            outputDTypes: [.uint32, .float32, x.dtype])
        return (outs[0].reshaped(1, M, topK), outs[1].reshaped(1, M, topK))
    }
}
