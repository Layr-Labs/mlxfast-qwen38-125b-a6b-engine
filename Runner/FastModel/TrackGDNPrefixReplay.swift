// CUDA's checkpoint/input-replay strategy adapted to MLX's request-owned
// prefix transaction. A verify retains compact inputs and one final SSM;
// only a strict accepted prefix reconstructs a state from its checkpoint.
import Foundation
import MLX
import MLXLMCommon

enum TrackGDNPrefixReplay {
    static let enabled = ProcessInfo.processInfo.environment["TRACK_GDN_PREFIX_REPLAY"] != "0"

    static func bytes(_ arrays: [MLXArray]) -> Int {
        arrays.reduce(0) { total, array in
            let (next, overflow) = total.addingReportingOverflow(array.nbytes)
            precondition(!overflow, "GDN replay retention overflow")
            return next
        }
    }

    static func forward(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray,
        geometry g: TrackFastKernels.GDNGeometry,
        evaluation: CBv2RecurrentStateEvaluation, layerIndex: Int
    ) throws -> MLXArray {
        let count = proj.dim(1)
        precondition(proj.dim(0) == 1 && (2...7).contains(count))
        let prep = TrackFastKernels.gdnPrep(
            proj: proj, convState: convState, convW: convW,
            negExpALog: negExpALog, dtBias: dtBias,
            T: count, capture: true, geometry: g)
        // Keep the existing two-value-row recurrence and its reduction order.
        // CAPTURE=false removes only the intermediate full-state stores.
        let rec = TrackFastKernels.leanTwoRowKernel(
            [prep[0], prep[1], prep[2], prep[3], prep[4], stateIn],
            template: [
                ("InT", proj.dtype), ("StT", stateIn.dtype), ("Dk", g.dk), ("Dv", g.dv),
                ("Hk", g.hk), ("Hv", g.hv), ("CAPTURE", false), ("T", count),
            ],
            grid: (32, g.dv / 2, g.hv), threadGroup: (32, 4, 1),
            outputShapes: [[1, count, g.hv, g.dv], [1, g.hv, g.dv, g.dk]],
            outputDTypes: [proj.dtype, stateIn.dtype])
        let k = prep[1], v = prep[2], decay = prep[3], beta = prep[4], conv = prep[5]
        let finalConv = conv[(count - 1)..<count]
        let finalSSM = rec[1]
        // Deliberately conservative: the input checkpoint may already be
        // charged to the committed generation, but is charged here as well.
        let retained = [k, v, decay, beta, conv, stateIn]
        try evaluation.stagePrefixReplay(
            modelLayerIndex: layerIndex, positions: count,
            finalConv: finalConv, finalSSM: finalSSM,
            materializedByteCount: bytes(retained + [finalSSM]),
            evaluationRoots: retained,
            strictReplayRetainedByteCount: bytes(retained),
            strictReplayRetainedRoots: retained,
            fullAcceptanceRetainedByteCount: conv.nbytes,
            fullAcceptanceRetainedRoots: [conv],
            replay: { keep in
                precondition(keep > 0 && keep < count)
                let state = replayKernel(
                    [k, v, decay, beta, stateIn],
                    template: [
                        ("InT", k.dtype), ("StT", stateIn.dtype),
                        ("Dk", g.dk), ("Dv", g.dv), ("Hk", g.hk), ("Hv", g.hv),
                        ("T", count), ("KEEP", keep),
                    ],
                    grid: (32, g.dv / 2, g.hv), threadGroup: (32, 4, 1),
                    outputShapes: [[1, g.hv, g.dv, g.dk]], outputDTypes: [stateIn.dtype])[0]
                return CBv2RecurrentLayerState(conv: conv[(keep - 1)..<keep], ssm: state)
            })
        return rec[0]
    }

    private static let replayKernel = MLXFast.metalKernel(
        name: "track_gdn_prefix_state_replay",
        inputNames: ["k", "v", "g", "beta", "state_in"],
        outputNames: ["state_out"], source: replaySource, ensureRowContiguous: true)

    // Same two-row state arithmetic as leanTwoRowSource. Query/output work
    // is absent because accepted-prefix materialization has no output reader.
    private static let replaySource = """
        const uint dv_idx = 2 * thread_position_in_grid.y;
        if (dv_idx >= Dv) { return; }
        const uint n = thread_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        const device InT* k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        const device InT* v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        const uint dk_idx = thread_position_in_threadgroup.x;
        const device float* g_ = g + b_idx * T * Hv;
        const device float* beta_ = beta + b_idx * T * Hv;
        const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
        float state0[n_per_t], state1[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            state0[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
            state1[i] = static_cast<float>(i_state[Dk + n_per_t * dk_idx + i]);
        }
        for (int t = 0; t < KEEP; ++t) {
            const float decay = g_[hv_idx];
            float kv_mem0 = 0.0f, kv_mem1 = 0.0f;
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_compensation0 = 0.0f, kv_compensation1 = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    const int s_idx = n_per_t * dk_idx + i;
                    const float key = static_cast<float>(k_[s_idx]);
                    {
                        state0[i] = state0[i] * decay;
                        auto product = state0[i] * key;
                        auto corrected = product - kv_compensation0;
                        auto next_sum = kv_mem0 + corrected;
                        kv_compensation0 = (next_sum - kv_mem0) - corrected;
                        kv_mem0 = next_sum;
                    }
                    {
                        state1[i] = state1[i] * decay;
                        auto product = state1[i] * key;
                        auto corrected = product - kv_compensation1;
                        auto next_sum = kv_mem1 + corrected;
                        kv_compensation1 = (next_sum - kv_mem1) - corrected;
                        kv_mem1 = next_sum;
                    }
                }
            }
            kv_mem0 = simd_sum(kv_mem0);
            kv_mem1 = simd_sum(kv_mem1);
            const float gate_beta = beta_[hv_idx];
            const float delta0 = (static_cast<float>(v_[dv_idx]) - kv_mem0) * gate_beta;
            const float delta1 = (static_cast<float>(v_[dv_idx + 1]) - kv_mem1) * gate_beta;
            for (int i = 0; i < n_per_t; ++i) {
                const int s_idx = n_per_t * dk_idx + i;
                const float key = static_cast<float>(k_[s_idx]);
                state0[i] = state0[i] + key * delta0;
                state1[i] = state1[i] + key * delta1;
            }
            k_ += Hk * Dk; v_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        device StT* o_state = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
            o_state[n_per_t * dk_idx + i] = static_cast<StT>(state0[i]);
            o_state[Dk + n_per_t * dk_idx + i] = static_cast<StT>(state1[i]);
        }
        """
}
