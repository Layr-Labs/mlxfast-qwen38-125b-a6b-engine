// Small-window GDN preparation, state transitions and gated normalization.
// Each threadgroup owns one value head and prepares every window position.
// Each SIMD group keeps four value rows across all steps. Capture stores the
// state and convolution history at every position in the existing slot layout.
// Arithmetic and BF16 conversions follow prepSource, leanTwoRowSource and gatedRMSSource.

import MLX

enum TrackFastGDNVerify {
    private static let kernel = MLXFast.metalKernel(
        name: "track_gdn_verify_complete",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "state_in", "w"],
        outputNames: ["state_out", "gated", "conv_out"],
        source: source, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func apply(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray, normW: MLXArray,
        zOffset: Int, eps: Float, capture: Bool, geometry g: TrackFastKernels.GDNGeometry
    ) -> (gated: MLXArray, stateOut: MLXArray, convOut: MLXArray)? {
        guard proj.ndim == 3, (1...8).contains(proj.dim(1)),
            capture || (2...4).contains(proj.dim(1)), proj.dtype == .bfloat16,
            stateIn.dtype == .float32, g.dk == 128, g.dv == 128,
            g.hk > 0, g.hv % g.hk == 0, g.convKernel > 1,
            g.convDim == (2 * g.hk + g.hv) * 128, g.projWidth == proj.dim(2),
            zOffset >= 0, zOffset + g.hv * g.dv <= g.projWidth,
            g.bOffset >= 0, g.bOffset + g.hv <= g.projWidth,
            g.aOffset >= 0, g.aOffset + g.hv <= g.projWidth
        else { return nil }
        let B = proj.dim(0), T = proj.dim(1)
        let slots = capture ? B * T : B
        guard convState.shape == [B, g.convKernel - 1, g.convDim],
            stateIn.shape == [B, g.hv, g.dv, g.dk],
            convW.shape == [g.convDim, g.convKernel],
            negExpALog.shape == [g.hv], dtBias.shape == [g.hv], normW.shape == [g.dv],
            convState.dtype == proj.dtype
        else { return nil }
        let result = kernel(
            [proj, convState, convW, negExpALog, dtBias, stateIn, normW],
            template: [
                ("InT", proj.dtype), ("StT", stateIn.dtype), ("Dk", g.dk), ("Dv", g.dv),
                ("Hk", g.hk), ("Hv", g.hv), ("KC", g.convKernel), ("CONV_DIM", g.convDim),
                ("PW", g.projWidth), ("B_OFF", g.bOffset), ("A_OFF", g.aOffset),
                ("Z_OFF", zOffset), ("EPS_BITS", Int(eps.bitPattern)), ("RPS", 4),
                ("T", T), ("CAPTURE", capture),
            ],
            grid: (32, g.dv / 4, B * g.hv), threadGroup: (32, g.dv / 4, 1),
            outputShapes: [[slots, g.hv, g.dv, g.dk], [B, T, g.hv * g.dv], [slots, g.convKernel - 1, g.convDim]],
            outputDTypes: [stateIn.dtype, proj.dtype, proj.dtype])
        return (result[1], result[0], result[2])
    }

    private static let source = #"""
        static_assert(Dk == 128 && Dv == 128 && T >= 1 && T <= 8 && RPS == 4, "small-window GDN geometry");
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        constexpr int KM1 = KC - 1;
        threadgroup InT q_shared[T][Dk];
        threadgroup InT k_shared[T][Dk];
        threadgroup InT v_shared[T][Dv];
        threadgroup float gb_shared[T][2];
        if (sg < 3 * T) {
            const uint t = sg / 3;
            const uint part = sg % 3;
            const uint vec = part == 0 ? hk_idx : (part == 1 ? Hk + hk_idx : 2 * Hk + hv_idx);
            const device InT* proj_b = proj + b_idx * T * PW;
            const device InT* cst_b = conv_state + b_idx * KM1 * CONV_DIM;
            auto win = [&](int r, uint ch) -> float {
                if (r < KM1) { return static_cast<float>(cst_b[(uint)(r * CONV_DIM) + ch]); }
                return static_cast<float>(proj_b[(uint)((r - KM1) * PW) + ch]);
            };
            float thread_x[4];
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const uint ch = vec * 128 + lane * 4 + i;
                float cacc = 0.0f;
                for (int j = 0; j < KC; ++j) {
                    cacc += win((int)t + j, ch) * conv_w[ch * KC + j];
                }
                const InT c0 = static_cast<InT>(cacc);
                const InT c1 = mlx_silu(c0);
                thread_x[i] = static_cast<float>(c1);
                acc += thread_x[i] * thread_x[i];
            }
            if (part < 2) {
                acc = simd_sum(acc);
                const float inv_mean = metal::precise::rsqrt(acc / 128.0f + 1e-6f);
                const float inv_scale = metal::rsqrt(static_cast<float>(Dk));
                const InT q_mul = static_cast<InT>(inv_scale * inv_scale);
                const InT k_mul = static_cast<InT>(inv_scale);
                for (int i = 0; i < 4; ++i) {
                    const InT normalized = static_cast<InT>(thread_x[i] * inv_mean);
                    const uint d = lane * 4 + i;
                    if (part == 0) { q_shared[t][d] = q_mul * normalized; }
                    else { k_shared[t][d] = k_mul * normalized; }
                }
            } else {
                for (int i = 0; i < 4; ++i) { v_shared[t][lane * 4 + i] = static_cast<InT>(thread_x[i]); }
            }
            if ((CAPTURE || t == (uint)(T - 1)) && (part == 2 || hv_idx % (Hv / Hk) == 0)) {
                const uint slot = CAPTURE ? b_idx * T + t : b_idx;
                device InT* o_conv = conv_out + slot * KM1 * CONV_DIM;
                for (int i = 0; i < 4; ++i) {
                    const uint ch = vec * 128 + lane * 4 + i;
                    for (int j = 0; j < KM1; ++j) {
                        o_conv[(uint)(j * CONV_DIM) + ch] = static_cast<InT>(win((int)t + 1 + j, ch));
                    }
                }
            }
        }
        if (sg >= 3 * T && sg < 4 * T && lane == 0) {
            const uint t = sg - 3 * T;
            const device InT* row = proj + (b_idx * T + t) * PW;
            const InT b_raw = row[B_OFF + hv_idx];
            gb_shared[t][1] = static_cast<float>(mlx_sigmoid(b_raw));
            const InT ax = row[A_OFF + hv_idx] + dt_bias[hv_idx];
            const InT sp = mlx_logaddexp0(ax);
            gb_shared[t][0] = metal::precise::exp(neg_exp_alog[hv_idx] * sp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float state[RPS][4];
        constexpr bool vec4 = metal::is_same<StT, float>::value;
        for (int r = 0; r < RPS; ++r) {
            const uint dv_idx = sg * RPS + r;
            const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
            if constexpr (vec4) {
                const float4 s4 = *reinterpret_cast<const device float4*>(i_state + 4 * lane);
                state[r][0] = s4.x; state[r][1] = s4.y; state[r][2] = s4.z; state[r][3] = s4.w;
            } else {
                for (int i = 0; i < 4; ++i) { state[r][i] = static_cast<float>(i_state[4 * lane + i]); }
            }
        }
        threadgroup InT y_shared[T][Dv];
        threadgroup float norm_sums[T][32];
        for (int t = 0; t < T; ++t) {
            const threadgroup InT* q_ = q_shared[t];
            const threadgroup InT* k_ = k_shared[t];
            const threadgroup InT* v_ = v_shared[t];
            const float gate_decay = gb_shared[t][0];
            const float gate_beta = gb_shared[t][1];
            for (int r = 0; r < RPS; ++r) {
                const uint dv_idx = sg * RPS + r;
                float kv_mem = 0.0f;
                {
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    float kv_compensation = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        const int s_idx = 4 * lane + i;
                        state[r][i] = state[r][i] * gate_decay;
                        auto product = state[r][i] * static_cast<float>(k_[s_idx]);
                        auto corrected = product - kv_compensation;
                        auto next_sum = kv_mem + corrected;
                        kv_compensation = (next_sum - kv_mem) - corrected;
                        kv_mem = next_sum;
                    }
                }
                kv_mem = simd_sum(kv_mem);
                const float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * gate_beta;
                float out = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const int s_idx = 4 * lane + i;
                    state[r][i] = state[r][i] + static_cast<float>(k_[s_idx]) * delta;
                    out += state[r][i] * static_cast<float>(q_[s_idx]);
                }
                out = simd_sum(out);
                if (lane == 0) {
                    const InT value = static_cast<InT>(out);
                    y_shared[t][dv_idx] = value;
                }
                if (CAPTURE || t == T - 1) {
                    const uint slot = CAPTURE ? b_idx * T + t : b_idx;
                    device StT* o_state = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;
                    if constexpr (vec4) {
                        *reinterpret_cast<device float4*>(o_state + 4 * lane) = float4(state[r][0], state[r][1], state[r][2], state[r][3]);
                    } else {
                        for (int i = 0; i < 4; ++i) { o_state[4 * lane + i] = static_cast<StT>(state[r][i]); }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float thread_x[4];
        if (sg < T) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                thread_x[i] = static_cast<float>(y_shared[sg][lane * 4 + i]);
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            norm_sums[sg][lane] = lane == 0 ? acc : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < T) {
            const float acc = simd_sum(norm_sums[sg][lane]);
            const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + as_type<float>((uint)EPS_BITS));
            for (int i = 0; i < 4; ++i) {
                const uint d = lane * 4 + i;
                InT normalized = w[d] * static_cast<InT>(thread_x[i] * inv_mean);
                const float z = static_cast<float>(proj[(b_idx * T + sg) * PW + Z_OFF + hv_idx * Dv + d]);
                const float zg = mlx_sigmoid(z);
                gated[((b_idx * T + sg) * Hv + hv_idx) * Dv + d] = static_cast<InT>(zg * static_cast<float>(normalized));
            }
        }
        """#
}
