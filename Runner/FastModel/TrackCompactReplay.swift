import Foundation
import MLX
import MLXLMCommon

/// One transaction-wide mode: every GDN and the PLE stage must opt in together.
/// The forward retains only the final SSM. A strict accepted prefix replays
/// the prepared recurrence inputs, without Q/output/normalization/projection.
enum TrackCompactReplay {
    static let enabled = ProcessInfo.processInfo.environment["TRACK_COMPACT_REPLAY"] != "0"

    private static func stateSource() -> String {
        var source = TrackFastKernels.leanTwoRowSource
        let replacements = [
            ("const device InT* q_ = q + b_idx * T_ * Hk * Dk + hk_idx * Dk;", ""),
            ("device InT* y_ = y + b_idx * T_ * Hv * Dv + hv_idx * Dv;", ""),
            ("float out0 = 0.0f, out1 = 0.0f;", ""),
            ("const float query = static_cast<float>(q_[s_idx]);", ""),
            ("out0 += state0[i] * query;", ""),
            ("out1 += state1[i] * query;", ""),
            ("out0 = simd_sum(out0);", ""),
            ("out1 = simd_sum(out1);", ""),
            ("y_[dv_idx] = static_cast<InT>(out0);", ""),
            ("y_[dv_idx + 1] = static_cast<InT>(out1);", ""),
            ("q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;",
             "k_ += Hk * Dk; v_ += Hv * Dv; g_ += Hv; beta_ += Hv;")
        ]
        for (old, new) in replacements {
            precondition(source.components(separatedBy: old).count == 2,
                         "TrackCompactReplay: recurrence anchor changed")
            source = source.replacingOccurrences(of: old, with: new)
        }
        return source
    }

    private static let stateKernel = MLXFast.metalKernel(
        name: "track_flash_prefix_state_only_v4",
        inputNames: ["k", "v", "g", "beta", "state_in"], outputNames: ["state_out"],
        source: stateSource(), ensureRowContiguous: true)

    static func bytes(_ arrays: [MLXArray]) -> Int {
        var total = 0
        for array in arrays {
            let (size, overflow1) = array.size.multipliedReportingOverflow(by: array.dtype.size)
            let (next, overflow2) = total.addingReportingOverflow(size)
            precondition(!overflow1 && !overflow2, "Flash replay byte accounting overflow")
            total = next
        }
        return total
    }

    static func gdn(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray,
        geometry: TrackFastKernels.GDNGeometry,
        evaluation: CBv2RecurrentStateEvaluation, layerIndex: Int
    ) throws -> MLXArray {
        let count = proj.dim(1), geo = geometry
        precondition(proj.dim(0) == 1 && (2 ... 8).contains(count))
        let prep = TrackFastKernels.gdnPrep(
            proj: proj, convState: convState, convW: convW, negExpALog: negExpALog,
            dtBias: dtBias, T: count, capture: true, geometry: geo)
        // Same two-row recurrence and arithmetic as captured verification;
        // only the per-position SSM stores are suppressed.
        let result = TrackFastKernels.leanTwoRowKernel(
            [prep[0], prep[1], prep[2], prep[3], prep[4], stateIn],
            template: [("InT", proj.dtype), ("StT", stateIn.dtype), ("Dk", geo.dk),
                       ("Dv", geo.dv), ("Hk", geo.hk), ("Hv", geo.hv),
                       ("CAPTURE", false), ("T", count)],
            grid: (32, geo.dv / 2, geo.hv), threadGroup: (32, 4, 1),
            outputShapes: [[1, count, geo.hv, geo.dv], [1, geo.hv, geo.dv, geo.dk]],
            outputDTypes: [proj.dtype, stateIn.dtype])
        let k = prep[1], v = prep[2], decay = prep[3], beta = prep[4], convs = prep[5]
        let finalState = result[1]
        let finalConv = convs[(count - 1) ..< count]
        let hadPrior = evaluation.inputState(modelLayerIndex: layerIndex)?.ssm != nil
        let tape = [k, v, decay, beta, convs]
        // The committed generation already charges an existing prior SSM.
        let pendingRoots = tape + (hadPrior ? [] : [stateIn])
        let strictRoots = tape + [stateIn]
        try evaluation.stagePrefixReplay(
            modelLayerIndex: layerIndex, positions: count,
            finalConv: finalConv, finalSSM: finalState,
            materializedByteCount: bytes(pendingRoots + [finalState]),
            evaluationRoots: strictRoots,
            strictReplayRetainedByteCount: bytes(strictRoots),
            strictReplayRetainedRoots: strictRoots,
            fullAcceptanceRetainedByteCount: bytes([convs]),
            fullAcceptanceRetainedRoots: [convs],
            replay: { keep in
                precondition(keep > 0 && keep < count)
                let ssm = stateKernel(
                    [k, v, decay, beta, stateIn],
                    template: [("InT", v.dtype), ("StT", stateIn.dtype), ("Dk", geo.dk),
                               ("Dv", geo.dv), ("Hk", geo.hk), ("Hv", geo.hv),
                               ("CAPTURE", false), ("T", keep)],
                    grid: (32, geo.dv / 2, geo.hv), threadGroup: (32, 4, 1),
                    outputShapes: [[1, geo.hv, geo.dv, geo.dk]], outputDTypes: [stateIn.dtype])[0]
                return CBv2RecurrentLayerState(conv: convs[(keep - 1) ..< keep], ssm: ssm)
            })
        return result[0]
    }

    /// PLE has a small convolution/context tape, not a large recurrent matrix.
    /// Retain its backing once and select the accepted position as a view.
    static func ple(
        evaluation: CBv2RecurrentStateEvaluation, layerIndex: Int, count: Int,
        convBacking: MLXArray, convStack: MLXArray,
        contextBacking: MLXArray, contextStack: MLXArray
    ) throws {
        let roots = [convBacking, contextBacking]
        let amount = bytes(roots)
        try evaluation.stagePrefixReplay(
            modelLayerIndex: layerIndex, positions: count,
            finalConv: convStack[(count - 1) ..< count],
            finalSSM: contextStack[(count - 1) ..< count],
            materializedByteCount: amount, evaluationRoots: roots,
            strictReplayRetainedByteCount: amount, strictReplayRetainedRoots: roots,
            fullAcceptanceRetainedByteCount: amount, fullAcceptanceRetainedRoots: roots,
            replay: { keep in
                precondition(keep > 0 && keep < count)
                return CBv2RecurrentLayerState(
                    conv: convStack[(keep - 1) ..< keep], ssm: contextStack[(keep - 1) ..< keep])
            })
    }
}
