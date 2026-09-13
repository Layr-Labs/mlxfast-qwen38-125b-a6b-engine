// TrackP12Prefill.swift -- prefill-only data-motion removals that keep every
// existing matrix call, tensor shape, reduction order and rounding point.
//
// WHAT THIS IS. The wide (prefill) forward already runs each projection of a
// `TrackMultiProj` as its own GEMM -- the split-K choice depends on N, so the
// fused row-concatenated weight is only used for windows of eight rows or
// fewer. What remains for wide windows is a pure *copy*: `concatenated(...)`
// materialises one wide buffer that the consumer kernels then read back with
// an offset. The same is true of the routed-expert path, which sorts the
// assignments, runs three sorted `gather_qmm` calls, and then scatters the
// result back into original slot order purely so that the combine kernel can
// read it with a linear address.
//
// Each change below removes one of those copies by *re-addressing* the
// consumer, not by changing what it computes:
//
//   * the gated-deltanet prep kernel reads the two 48-wide gate projections
//     from their own buffers instead of from offsets in a 16480-wide
//     concatenation, and the gated RMS kernel reads the z projection from its
//     own buffer at offset 0;
//   * the MoE combine kernel reads the sorted expert rows through the inverse
//     permutation the sort already produced, instead of reading a scattered
//     copy of them;
//   * the shared expert uses the two-input SwiGLU kernel over the separate
//     gate and up outputs instead of the fused-layout one over their
//     concatenation;
//   * the attention QKV concatenation drops its fourth part, the narrow
//     indexer-K projection, which wide windows do not read (they take the
//     indexer tape from the full `index_qk_proj` instead, because the GEMM
//     rounding there depends on N).
//
// EXACTNESS. The two re-addressed Metal kernels are generated from the
// original kernel sources by textual substitution of the address expression
// alone, with a precondition that the anchor still occurs exactly once, so a
// later edit of the original kernel fails the build instead of silently
// diverging. The SwiGLU kernels compute `mlx_silu(gate) * up` in both
// layouts. Every GEMM keeps its own M/N/K and therefore its own split-K
// decision and accumulation order; no weight is re-quantised, re-ordered or
// re-associated, and the routed combine keeps the float32 products in
// original slot order and the same small-column reduction tree.
//
// Each removal is separately switchable at process start (see below) so the
// paired benchmark can attribute its own measurement.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum TrackP12Prefill {
    // MARK: - Switches (all default ON; set a variable to "0" to fall back)

    /// `TRACK_P12_EXACT_PREFILL=0` restores every pre-change path at once.
    private static let master =
        ProcessInfo.processInfo.environment["TRACK_P12_EXACT_PREFILL"] != "0"

    private static func on(_ name: String) -> Bool {
        master && ProcessInfo.processInfo.environment[name] != "0"
    }

    static let splitGDN = on("TRACK_P12_SPLIT_GDN_INPUTS")
    static let sortedCombine = on("TRACK_P12_SORTED_COMBINE")
    static let splitShared = on("TRACK_P12_SPLIT_SHARED_INPUTS")
    static let omitUnusedIndexer = on("TRACK_P12_OMIT_UNUSED_INDEXER")
    static let splitAttention = on("TRACK_P12_SPLIT_ATTN_INPUTS")

    /// Wide, batch-one, activation-dtype windows only: the decode and verify
    /// windows keep the fused projections and the small-window MoE kernels.
    static func eligible(_ x: MLXArray) -> Bool {
        x.ndim == 3 && x.dim(0) == 1 && x.dim(1) > 8 && x.dtype == .bfloat16
    }

    /// Reuse an existing kernel body literally, replacing only the listed
    /// address expressions. Fails closed if an anchor is no longer unique.
    private static func addressVariant(
        _ source: String, _ replacements: [(String, String)]
    ) -> String {
        var result = source
        for (old, new) in replacements {
            precondition(
                result.components(separatedBy: old).count == 2,
                "TrackP12Prefill: kernel anchor '\(old)' is no longer unique; re-audit this variant")
            result = result.replacingOccurrences(of: old, with: new)
        }
        return result
    }

    // MARK: - 1. Gated-deltanet input projections without the concatenation

    /// `track_gdn_prep` with the beta/alpha gate rows read from their own
    /// projection outputs. Everything else -- threads, reductions, casts, the
    /// convolution window, the conv-tail store -- is the original source, and
    /// `PROJ_W` is now the QKV row pitch (10240) rather than the concatenated
    /// one (16480).
    nonisolated(unsafe) private static let splitPrepKernel = MLXFast.metalKernel(
        name: "track_p12_gdn_prep_split_inputs",
        inputNames: [
            "proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "b_gate", "a_gate",
        ],
        outputNames: ["qn", "kn", "vv", "g", "beta", "conv_out"],
        source: addressVariant(
            TrackFastKernels.prepSource,
            [
                ("row[B_OFF + hh]", "b_gate[bt * Hv + hh]"),
                ("row[A_OFF + hh]", "a_gate[bt * Hv + hh]"),
            ]),
        header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func gdnPrepSplit(
        proj: MLXArray, b: MLXArray, a: MLXArray,
        convState: MLXArray, convW: MLXArray, negExpALog: MLXArray,
        dtBias: MLXArray, T: Int, capture: Bool, geometry g: TrackFastKernels.GDNGeometry
    ) -> [MLXArray] {
        let B = proj.dim(0)
        let slots = capture ? B * T : B
        precondition(g.projWidth == g.convDim && proj.dim(2) == g.convDim)
        precondition(b.dim(2) == g.hv && a.dim(2) == g.hv)
        precondition(b.dtype == proj.dtype && a.dtype == proj.dtype)
        precondition(g.dk == 128 && g.dv == 128 && g.convDim % 128 == 0)
        return splitPrepKernel(
            [proj, convState, convW, negExpALog, dtBias, b, a],
            template: [
                ("InT", proj.dtype), ("T", T), ("Dk", g.dk), ("Dv", g.dv), ("Hk", g.hk),
                ("Hv", g.hv), ("KC", g.convKernel), ("PROJ_W", g.projWidth),
                ("CONV_DIM", g.convDim), ("B_OFF", g.bOffset), ("A_OFF", g.aOffset),
                ("CAPTURE", capture),
            ],
            grid: (32, g.convDim / 128, B * T), threadGroup: (32, 4, 1),
            outputShapes: [
                [B, T, g.hk, g.dk], [B, T, g.hk, g.dk], [B, T, g.hv, g.dv],
                [B, T, g.hv], [B, T, g.hv], [slots, g.convKernel - 1, g.convDim],
            ],
            outputDTypes: [proj.dtype, proj.dtype, proj.dtype, .float32, .float32, proj.dtype])
    }

    /// The same geometry with the QKV pitch in place of the concatenated one.
    static func splitGeometry(_ g: TrackFastKernels.GDNGeometry)
        -> TrackFastKernels.GDNGeometry
    {
        TrackFastKernels.GDNGeometry(
            projWidth: g.convDim, convDim: g.convDim, convKernel: g.convKernel,
            hk: g.hk, hv: g.hv, dk: g.dk, dv: g.dv, bOffset: g.bOffset, aOffset: g.aOffset)
    }

    nonisolated(unsafe) private static let splitAttnPrepKernel = MLXFast.metalKernel(
        name: "track_p12_attn_prep_split_inputs",
        inputNames: ["qkv", "kproj", "vproj", "qnorm", "knorm", "cosb", "sinb"],
        outputNames: ["qout", "kout", "vout"],
        source: addressVariant(
            TrackFastKernels.attnPrepSource,
            [
                ("src = row * QW + 2 * HQ * D + hh * D;", "src = row * HK * D + hh * D;"),
                ("src = row * QW + 2 * HQ * D + HK * D + hh * D;", "src = row * HK * D + hh * D;"),
                ("qkv[src + d]", "vproj[src + d]"),
                ("qkv[src + lid * N_READS + i]", "(isQ ? qkv : kproj)[src + lid * N_READS + i]"),
            ]),
        header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func attnPrepSplit(
        qGate: MLXArray, k: MLXArray, v: MLXArray,
        qNorm: MLXArray, kNorm: MLXArray, cos: MLXArray, sin: MLXArray,
        heads: Int, kvHeads: Int, headDim: Int, rotaryDims: Int, eps: Float
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let B = qGate.dim(0), S = qGate.dim(1)
        precondition(headDim % 4 == 0 && rotaryDims % 8 == 0 && cos.dim(1) == rotaryDims)
        precondition(qGate.dim(2) == 2 * heads * headDim)
        precondition(k.shape == [B, S, kvHeads * headDim] && v.shape == k.shape)
        precondition(k.dtype == qGate.dtype && v.dtype == qGate.dtype)
        let outs = splitAttnPrepKernel(
            [qGate, k, v, qNorm, kNorm, cos, sin],
            template: [
                ("InT", qGate.dtype), ("D", headDim), ("HQ", heads), ("HK", kvHeads), ("S", S),
                ("QW", qGate.dim(2)), ("ROT", rotaryDims), ("EPS_BITS", Int(eps.bitPattern)),
            ],
            grid: (headDim / 4, heads + 2 * kvHeads, B * S), threadGroup: (headDim / 4, 1, 1),
            outputShapes: [[B, heads, S, headDim], [B, kvHeads, S, headDim], [B, kvHeads, S, headDim]],
            outputDTypes: [qGate.dtype, qGate.dtype, qGate.dtype])
        return (outs[0], outs[1], outs[2])
    }

    // MARK: - 2. Routed experts: combine straight out of the sorted rows

    /// `track_moe_combine` with the routed row address taken through the
    /// inverse permutation the expert sort already produced. The float32
    /// products stay in original (token, slot) order and the K reduction is
    /// still the same small-column tree -- only the load address changes.
    nonisolated(unsafe) private static let sortedCombineKernel = MLXFast.metalKernel(
        name: "track_p12_moe_sorted_combine",
        inputNames: ["routed", "w", "shared", "gate", "inverse_order"],
        outputNames: ["out"],
        source: addressVariant(
            TrackFastKernels.moeCombineSource,
            [
                (
                    "routed[(row * K + k) * H + d]",
                    "routed[static_cast<uint>(inverse_order[row * K + k]) * H + d]"
                )
            ]),
        header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    /// Keep sorted expert rows through the projections and weighted combine.
    static func sortedMoE(
        _ m: TrackMoE, _ x: MLXArray, indices: MLXArray, weights: MLXArray,
        shared: MLXArray, gate: MLXArray
    ) -> MLXArray? {
        guard sortedCombine, eligible(x), indices.size >= 64, weights.dtype == .float32,
            let parts = m.p12SortedParts, !m.switchMLP.hasFusedGateUp
        else { return nil }
        let B = x.dim(0), S = x.dim(1), H = x.dim(2), K = indices.dim(-1)
        let activated: MLXArray
        let sortedIDs: MLXArray
        let inverse: MLXArray
        if let indirect = TrackPrefillIndirect.apply(m, x: x, indices: indices) {
            (activated, sortedIDs, inverse) = indirect
        } else {
            let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            let sorted = gatherSort(x: expanded, indices: indices)
            sortedIDs = sorted.1
            inverse = sorted.2
            let up = parts.up(sorted.0, sortedIDs, sortedIndices: true)
            let gateAct = parts.gate(sorted.0, sortedIDs, sortedIndices: true)
            activated = compiledSiluProduct(gateAct, up)
        }
        let down = parts.down(activated, sortedIDs, sortedIndices: true)
        guard down.ndim == 3, down.dim(0) == B * S * K, down.dim(1) == 1, down.dim(2) == H,
            inverse.size == B * S * K
        else { return nil }
        return sortedCombineKernel(
            [down, weights, shared, gate, inverse],
            template: [("InT", down.dtype), ("K", K), ("H", H)],
            grid: (H, B * S, 1), threadGroup: (256, 1, 1),
            outputShapes: [[B, S, H]], outputDTypes: [down.dtype])[0]
    }
}
