// Compact MTP recurrent capture: final state plus forward-rounded innovations.
// Derived from the promoted TrackFastKernels prep and two-row recurrence.
// No target weights or arithmetic reduction order are changed.
import MLX
import MLXLMCommon

enum TrackCompactReplay {
    static func byteCount(_ roots: [MLXArray]) -> Int {
        var total = 0
        for root in roots {
            let (bytes, mulOverflow) = root.size.multipliedReportingOverflow(by: root.dtype.size)
            let (next, addOverflow) = total.addingReportingOverflow(bytes)
            precondition(!mulOverflow && !addOverflow, "compact replay byte count overflow")
            total = next
        }
        return total
    }

    // Copy a dense window into independent storage; no floating-point addition,
    // so signed zeros and every source bit are preserved in committed tails.
    nonisolated(unsafe) static let tailKernel = MLXFast.metalKernel(
        name: "track_compact_tail", inputNames: ["history"], outputNames: ["tail"],
        source: """
            const uint i = thread_position_in_grid.x;
            if (i < COUNT) { tail[i] = history[OFFSET + i]; }
            """, ensureRowContiguous: true)

    static func tail(_ history: MLXArray, offset: Int, shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        precondition(offset >= 0 && offset + count <= history.size)
        return tailKernel(
            [history], template: [("COUNT", count), ("OFFSET", offset)],
            grid: (count, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [shape], outputDTypes: [history.dtype])[0]
    }

    static let prepSource = """
        constexpr int KM1 = KC - 1;
        constexpr int N_READS = 4;
        constexpr int VEC_Q = Hk;
        constexpr int VEC_K = 2 * Hk;
        const uint lane = thread_position_in_threadgroup.x;   // 0..31
        const uint vec = thread_position_in_grid.y;           // vector index
        const uint bt = thread_position_in_grid.z;            // b*T + t
        const uint b = bt / T;
        const uint t = bt % T;
        const device InT* proj_b = proj + (uint)(b * T * PROJ_W);
        const device InT* cst_b = conv_state + (uint)(b * KM1 * CONV_DIM);
        auto win = [&](int r, uint ch) -> float {
            if (r < KM1) { return static_cast<float>(cst_b[(uint)(r * CONV_DIM) + ch]); }
            return static_cast<float>(proj_b[(uint)((r - KM1) * PROJ_W) + ch]);
        };
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint ch = vec * 128 + lane * N_READS + i;
            float cacc = 0.0f;
            for (int j = 0; j < KC; ++j) {
                cacc += win((int)t + j, ch) * conv_w[ch * KC + j];
            }
            const InT c0 = static_cast<InT>(cacc);
            const InT c1 = mlx_silu(c0);
            thread_x[i] = static_cast<float>(c1);
            acc += thread_x[i] * thread_x[i];
        }
        if (vec < VEC_K) {
            acc = simd_sum(acc);
            const float inv_mean = metal::precise::rsqrt(acc / 128.0f + 1e-6f);
            const float inv_scale = metal::rsqrt(static_cast<float>(Dk));
            const InT q_mul = static_cast<InT>(inv_scale * inv_scale);
            const InT k_mul = static_cast<InT>(inv_scale);
            for (int i = 0; i < N_READS; ++i) {
                const InT n = static_cast<InT>(thread_x[i] * inv_mean);
                const uint d = lane * N_READS + i;
                if (vec < VEC_Q) {
                    qn[(uint)((bt * Hk + vec) * Dk) + d] = q_mul * n;
                } else {
                    kn[(uint)((bt * Hk + (vec - VEC_Q)) * Dk) + d] = k_mul * n;
                }
            }
        } else {
            for (int i = 0; i < N_READS; ++i) {
                const uint d = lane * N_READS + i;
                vv[(uint)((bt * Hv + (vec - VEC_K)) * Dv) + d] = static_cast<InT>(thread_x[i]);
            }
        }
        if (vec == 0) {
            const device InT* row = proj_b + (uint)(t * PROJ_W);
            for (int hh = lane; hh < Hv; hh += 32) {
                const InT b_raw = row[B_OFF + hh];
                beta[bt * Hv + hh] = static_cast<float>(mlx_sigmoid(b_raw));
                const InT ax = row[A_OFF + hh] + dt_bias[hh];
                const InT sp = mlx_logaddexp0(ax);
                g[bt * Hv + hh] = metal::precise::exp(neg_exp_alog[hh] * sp);
            }
        }
        if (CAPTURE || t == (uint)(T - 1)) {
            const uint slot = CAPTURE ? bt : b;
            device InT* o_conv = conv_out + (uint)(slot * KM1 * CONV_DIM);
            for (int i = 0; i < N_READS; ++i) {
                const uint ch = vec * 128 + lane * N_READS + i;
                for (int j = 0; j < KM1; ++j) {
                    o_conv[(uint)(j * CONV_DIM) + ch] = static_cast<InT>(win((int)t + 1 + j, ch));
                }
            }
        }
        // Each (batch, token, channel) writes exactly one new history row.
        // Token zero also copies the committed convolution prefix once.
        for (int i = 0; i < N_READS; ++i) {
            const uint ch = vec * 128 + lane * N_READS + i;
            const uint base = b * (KM1 + T) * CONV_DIM;
            conv_history[base + (KM1 + t) * CONV_DIM + ch] =
                proj_b[t * PROJ_W + ch];
            if (t == 0) {
                for (int j = 0; j < KM1; ++j) {
                    conv_history[base + j * CONV_DIM + ch] = cst_b[j * CONV_DIM + ch];
                }
            }
        }
        """

    static let forwardSource = """
        // MLXFAST-GDNTILE: Each row retains four consecutive key elements per lane.
        const uint dv_idx = 2 * thread_position_in_grid.y;
        if (dv_idx >= Dv) { return; }
        const uint n = thread_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        const int T_ = T;
        const device InT* q_ = q + b_idx * T_ * Hk * Dk + hk_idx * Dk;
        const device InT* k_ = k + b_idx * T_ * Hk * Dk + hk_idx * Dk;
        const device InT* v_ = v + b_idx * T_ * Hv * Dv + hv_idx * Dv;
        device InT* y_ = y + b_idx * T_ * Hv * Dv + hv_idx * Dv;
        const uint dk_idx = thread_position_in_threadgroup.x;
        const device float* g_ = g + b_idx * T_ * Hv;
        const device float* beta_ = beta + b_idx * T_ * Hv;
        const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
        float state0[n_per_t], state1[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            state0[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
            state1[i] = static_cast<float>(i_state[Dk + n_per_t * dk_idx + i]);
        }
        for (int t = 0; t < T_; ++t) {
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
            float out0 = 0.0f, out1 = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                const int s_idx = n_per_t * dk_idx + i;
                const float key = static_cast<float>(k_[s_idx]);
                state0[i] = state0[i] + key * delta0;
                state1[i] = state1[i] + key * delta1;
                const float query = static_cast<float>(q_[s_idx]);
                out0 += state0[i] * query;
                out1 += state1[i] * query;
            }
            out0 = simd_sum(out0);
            out1 = simd_sum(out1);
            if (dk_idx == 0) {
                const uint log_row = ((b_idx * T_ + t) * Hv + hv_idx) * Dv;
                delta_log[log_row + dv_idx] = delta0;
                delta_log[log_row + dv_idx + 1] = delta1;
                y_[dv_idx] = static_cast<InT>(out0);
                y_[dv_idx + 1] = static_cast<InT>(out1);
            }
            if (t == T_ - 1) {
                const uint slot = b_idx;
                device StT* o_state = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;
                for (int i = 0; i < n_per_t; ++i) {
                    o_state[n_per_t * dk_idx + i] = static_cast<StT>(state0[i]);
                    o_state[Dk + n_per_t * dk_idx + i] = static_cast<StT>(state1[i]);
                }
            }
            q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        """

    nonisolated(unsafe) static let prepKernel = MLXFast.metalKernel(
        name: "track_compact_gdn_prep",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias"],
        outputNames: ["qn", "kn", "vv", "g", "beta", "conv_out", "conv_history"],
        source: prepSource, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    nonisolated(unsafe) static let forwardKernel = MLXFast.metalKernel(
        name: "track_compact_gdn_forward",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out", "delta_log"],
        source: forwardSource, ensureRowContiguous: true)

    // The decay keeps the original non-contracting scope; the rank-one update
    // keeps the original expression and default contraction policy. No dot or
    // reduction is recomputed: delta_log contains the actual forward FP32 delta.
    static let replaySource = """
        const uint dv_idx = 2 * thread_position_in_grid.y;
        if (dv_idx >= Dv) { return; }
        const uint hv_idx = thread_position_in_grid.z;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint dk_idx = thread_position_in_threadgroup.x;
        constexpr int n_per_t = Dk / 32;
        const device StT* i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
        float state0[n_per_t], state1[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            state0[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
            state1[i] = static_cast<float>(i_state[Dk + n_per_t * dk_idx + i]);
        }
        for (int t = 0; t < KEEP; ++t) {
            const float decay = g[t * Hv + hv_idx];
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                for (int i = 0; i < n_per_t; ++i) {
                    state0[i] = state0[i] * decay;
                    state1[i] = state1[i] * decay;
                }
            }
            const uint log_row = (t * Hv + hv_idx) * Dv;
            const float delta0 = delta_log[log_row + dv_idx];
            const float delta1 = delta_log[log_row + dv_idx + 1];
            const device InT* k_ = k + (t * Hk + hk_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
                const int s_idx = n_per_t * dk_idx + i;
                const float key = static_cast<float>(k_[s_idx]);
                state0[i] = state0[i] + key * delta0;
                state1[i] = state1[i] + key * delta1;
            }
        }
        device StT* o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
            o_state[n_per_t * dk_idx + i] = static_cast<StT>(state0[i]);
            o_state[Dk + n_per_t * dk_idx + i] = static_cast<StT>(state1[i]);
        }
        """

    nonisolated(unsafe) static let replayKernel = MLXFast.metalKernel(
        name: "track_compact_gdn_replay",
        inputNames: ["k", "g", "delta_log", "state_in"], outputNames: ["state_out"],
        source: replaySource, ensureRowContiguous: true)

    static func gdn(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray,
        geometry geo: TrackFastKernels.GDNGeometry,
        evaluation: CBv2RecurrentStateEvaluation, layerIndex: Int
    ) throws -> MLXArray {
        let width = proj.dim(1)
        precondition(proj.dim(0) == 1 && width >= 2)
        precondition(geo.dk == 128 && geo.dv == 128 && geo.convDim % 128 == 0)
        precondition(stateIn.dtype == .float32)
        let prep = prepKernel(
            [proj, convState, convW, negExpALog, dtBias],
            template: [
                ("InT", proj.dtype), ("T", width), ("Dk", geo.dk), ("Dv", geo.dv),
                ("Hk", geo.hk), ("Hv", geo.hv), ("KC", geo.convKernel),
                ("PROJ_W", geo.projWidth), ("CONV_DIM", geo.convDim),
                ("B_OFF", geo.bOffset), ("A_OFF", geo.aOffset), ("CAPTURE", false),
            ],
            grid: (32, geo.convDim / 128, width), threadGroup: (32, 4, 1),
            outputShapes: [
                [1, width, geo.hk, geo.dk], [1, width, geo.hk, geo.dk],
                [1, width, geo.hv, geo.dv], [1, width, geo.hv], [1, width, geo.hv],
                [1, geo.convKernel - 1, geo.convDim],
                [1, geo.convKernel - 1 + width, geo.convDim],
            ],
            outputDTypes: [proj.dtype, proj.dtype, proj.dtype, .float32, .float32,
                          proj.dtype, proj.dtype])
        let rec = forwardKernel(
            [prep[0], prep[1], prep[2], prep[3], prep[4], stateIn],
            template: [
                ("InT", proj.dtype), ("StT", stateIn.dtype), ("Dk", geo.dk), ("Dv", geo.dv),
                ("Hk", geo.hk), ("Hv", geo.hv), ("T", width),
            ],
            grid: (32, geo.dv / 2, geo.hv), threadGroup: (32, 4, 1),
            outputShapes: [[1, width, geo.hv, geo.dv], [1, geo.hv, geo.dv, geo.dk],
                           [1, width, geo.hv, geo.dv]],
            outputDTypes: [proj.dtype, stateIn.dtype, .float32])
        let key = prep[1], decay = prep[3], history = prep[6], delta = rec[2]
        let finalConv = prep[5], finalSSM = rec[1]
        let logs = [key, decay, delta, history]
        let retained = logs + [stateIn]
        // Existing committed state is already charged during verification;
        // newly initialized state must be included in the pending charge.
        let initial = evaluation.inputState(modelLayerIndex: layerIndex)?.ssm == nil
            ? [stateIn] : []
        try evaluation.stagePrefixReplay(
            modelLayerIndex: layerIndex, positions: width,
            finalConv: finalConv, finalSSM: finalSSM,
            materializedByteCount: byteCount(logs + initial + [finalConv, finalSSM]),
            evaluationRoots: retained,
            strictReplayRetainedByteCount: byteCount(retained),
            strictReplayRetainedRoots: retained,
            replay: { keep in
                precondition(keep > 0 && keep < width)
                let ssm = replayKernel(
                    [key, decay, delta, stateIn],
                    template: [
                        ("InT", key.dtype), ("StT", stateIn.dtype), ("Dk", geo.dk),
                        ("Dv", geo.dv), ("Hk", geo.hk), ("Hv", geo.hv), ("KEEP", keep),
                    ],
                    grid: (32, geo.dv / 2, geo.hv), threadGroup: (32, 4, 1),
                    outputShapes: [[1, geo.hv, geo.dv, geo.dk]],
                    outputDTypes: [stateIn.dtype])[0]
                let conv = tail(history, offset: keep * geo.convDim,
                                shape: [1, geo.convKernel - 1, geo.convDim])
                return CBv2RecurrentLayerState(conv: conv, ssm: ssm)
            })
        return rec[0]
    }

    static func stagePLE(
        full: MLXArray, history: MLXArray, positions: Int, stateLength: Int,
        contextLength: Int, wide: Int, layerIndex: Int,
        evaluation: CBv2RecurrentStateEvaluation
    ) throws {
        precondition(positions >= 2 && full.dim(0) == 1 && history.dtype == .int32)
        let roots = [full, history]
        let convShape = [1, stateLength, wide]
        let contextShape = [1, contextLength]
        let finalConv = tail(full, offset: positions * wide, shape: convShape)
        let finalSSM = tail(history, offset: positions, shape: contextShape)
        try evaluation.stagePrefixReplay(
            modelLayerIndex: layerIndex, positions: positions,
            finalConv: finalConv, finalSSM: finalSSM,
            materializedByteCount: byteCount(roots + [finalConv, finalSSM]),
            evaluationRoots: roots,
            strictReplayRetainedByteCount: byteCount(roots),
            strictReplayRetainedRoots: roots,
            replay: { keep in
                precondition(keep > 0 && keep < positions)
                return CBv2RecurrentLayerState(
                    conv: tail(full, offset: keep * wide, shape: convShape),
                    ssm: tail(history, offset: keep, shape: contextShape))
            })
    }
}
