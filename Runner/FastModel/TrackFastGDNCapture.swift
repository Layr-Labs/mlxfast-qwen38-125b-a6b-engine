// Two-position capture GDN preparation, recurrence and gated normalization.
// Each threadgroup owns one value head; each SIMD group owns four value rows.
// Both recurrent and convolution outputs retain every prefix for rollback.
// Per-row arithmetic, intermediate BF16 conversions and reduction lanes follow
// TrackFastKernels.prepSource, leanSource and gatedRMSSource.

import MLX

enum TrackFastGDNCapture {
    private static let kernel = MLXFast.metalKernel(
        name: "track_gdn_capture_pair_complete",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "state_in", "w"],
        outputNames: ["state_out", "gated", "conv_out"],
        source: source, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func apply(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray, normW: MLXArray,
        zOffset: Int, eps: Float, capture: Bool, geometry g: TrackFastKernels.GDNGeometry
    ) -> (gated: MLXArray, stateOut: MLXArray, convOut: MLXArray)? {
        guard capture, proj.ndim == 3, proj.dim(1) == 2, proj.dtype == .bfloat16,
            stateIn.dtype == .float32, g.dk == 128, g.dv == 128,
            g.hk > 0, g.hv % g.hk == 0, g.convKernel > 1,
            g.convDim == (2 * g.hk + g.hv) * 128, g.projWidth == proj.dim(2),
            zOffset >= 0, zOffset + g.hv * g.dv <= g.projWidth,
            g.bOffset >= 0, g.bOffset + g.hv <= g.projWidth,
            g.aOffset >= 0, g.aOffset + g.hv <= g.projWidth
        else { return nil }
        let B = proj.dim(0), T = proj.dim(1)
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
                ("T", T), ("Z_OFF", zOffset), ("EPS_BITS", Int(eps.bitPattern)), ("RPS", 4),
            ],
            grid: (32, g.dv / 4, B * g.hv), threadGroup: (32, g.dv / 4, 1),
            outputShapes: [[B * T, g.hv, g.dv, g.dk], [B, T, g.hv * g.dv], [B * T, g.convKernel - 1, g.convDim]],
            outputDTypes: [stateIn.dtype, proj.dtype, proj.dtype])
        return (result[1], result[0], result[2])
    }

    private static let source = #"""
        static_assert(T == 2 && Dk == 128 && Dv == 128, "GDN pair capture geometry");
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        constexpr int KM1 = KC - 1;
        threadgroup InT q_shared[T * Dk];
        threadgroup InT k_shared[T * Dk];
        threadgroup InT v_shared[T * Dv];
        threadgroup float gb_shared[T * 2];
        float4 next_state;
        if constexpr (metal::is_same<StT, float>::value) {
            const device StT* first_state = state_in + (n * Dv + sg * RPS) * Dk;
            next_state = *reinterpret_cast<const device float4*>(first_state + 4 * lane);
        }
        for (int t = 0; t < T; ++t) {
            if (sg < 3) {
                const uint vec = sg == 0 ? hk_idx : (sg == 1 ? Hk + hk_idx : 2 * Hk + hv_idx);
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
                        cacc += win(t + j, ch) * conv_w[ch * KC + j];
                    }
                    const InT c0 = static_cast<InT>(cacc);
                    const InT c1 = mlx_silu(c0);
                    thread_x[i] = static_cast<float>(c1);
                    acc += thread_x[i] * thread_x[i];
                }
                if (sg < 2) {
                    acc = simd_sum(acc);
                    const float inv_mean = metal::precise::rsqrt(acc / 128.0f + 1e-6f);
                    const float inv_scale = metal::rsqrt(static_cast<float>(Dk));
                    const InT q_mul = static_cast<InT>(inv_scale * inv_scale);
                    const InT k_mul = static_cast<InT>(inv_scale);
                    for (int i = 0; i < 4; ++i) {
                        const InT normalized = static_cast<InT>(thread_x[i] * inv_mean);
                        const uint d = lane * 4 + i;
                        if (sg == 0) { q_shared[t * Dk + d] = q_mul * normalized; }
                        else { k_shared[t * Dk + d] = k_mul * normalized; }
                    }
                } else {
                    for (int i = 0; i < 4; ++i) { v_shared[t * Dv + lane * 4 + i] = static_cast<InT>(thread_x[i]); }
                }
                if (sg == 2 || hv_idx % (Hv / Hk) == 0) {
                    device InT* o_conv = conv_out + (b_idx * T + t) * KM1 * CONV_DIM;
                    for (int i = 0; i < 4; ++i) {
                        const uint ch = vec * 128 + lane * 4 + i;
                        for (int j = 0; j < KM1; ++j) {
                            o_conv[(uint)(j * CONV_DIM) + ch] = static_cast<InT>(win(t + 1 + j, ch));
                        }
                    }
                }
            }
            if (sg == 0 && lane == 0) {
                const device InT* row = proj + (b_idx * T + t) * PW;
                const InT b_raw = row[B_OFF + hv_idx];
                gb_shared[t * 2 + 1] = static_cast<float>(mlx_sigmoid(b_raw));
                const InT ax = row[A_OFF + hv_idx] + dt_bias[hv_idx];
                const InT sp = mlx_logaddexp0(ax);
                gb_shared[t * 2] = metal::precise::exp(neg_exp_alog[hv_idx] * sp);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup InT y_shared[T * Dv];
        for (int r = 0; r < RPS; ++r) {
            const uint dv_idx = sg * RPS + r;
            const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
            float state[4];
            constexpr bool vec4 = metal::is_same<StT, float>::value;
            if constexpr (vec4) {
                const float4 s4 = next_state;
                if (r + 1 < RPS) {
                    next_state = *reinterpret_cast<const device float4*>(i_state + Dk + 4 * lane);
                }
                state[0] = s4.x; state[1] = s4.y; state[2] = s4.z; state[3] = s4.w;
            } else {
                for (int i = 0; i < 4; ++i) { state[i] = static_cast<float>(i_state[4 * lane + i]); }
            }
            for (int t = 0; t < T; ++t) {
                const threadgroup InT* q_ = q_shared + t * Dk;
                const threadgroup InT* k_ = k_shared + t * Dk;
                const threadgroup InT* v_ = v_shared + t * Dv;
                const float gate_decay = gb_shared[t * 2];
                const float gate_beta = gb_shared[t * 2 + 1];
                const float4 local_q = float4(
                    static_cast<float>(q_[4 * lane]), static_cast<float>(q_[4 * lane + 1]),
                    static_cast<float>(q_[4 * lane + 2]), static_cast<float>(q_[4 * lane + 3]));
                const float4 local_k = float4(
                    static_cast<float>(k_[4 * lane]), static_cast<float>(k_[4 * lane + 1]),
                    static_cast<float>(k_[4 * lane + 2]), static_cast<float>(k_[4 * lane + 3]));
                float kv_mem;
                {
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    state[0] = state[0] * gate_decay;
                    kv_mem = 0.0f + state[0] * local_k[0];
                    float kv_compensation = 0.0f;
                    for (int i = 1; i < 4; ++i) {
                        state[i] = state[i] * gate_decay;
                        auto product = state[i] * local_k[i];
                        auto corrected = product - kv_compensation;
                        auto next_sum = kv_mem + corrected;
                        if (i + 1 < 4) { kv_compensation = (next_sum - kv_mem) - corrected; }
                        kv_mem = next_sum;
                    }
                }
                kv_mem = simd_sum(kv_mem);
                const float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * gate_beta;
                float out = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    state[i] = state[i] + local_k[i] * delta;
                    out += state[i] * local_q[i];
                }
                out = simd_sum(out);
                if (lane == 0) {
                    const InT value = static_cast<InT>(out);
                    y_shared[t * Dv + dv_idx] = value;
                }
                device StT* o_state = state_out + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
                if constexpr (vec4) {
                    *reinterpret_cast<device float4*>(o_state + 4 * lane) = float4(state[0], state[1], state[2], state[3]);
                } else {
                    for (int i = 0; i < 4; ++i) { o_state[4 * lane + i] = static_cast<StT>(state[i]); }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (int t = 0; t < T; ++t) {
                float thread_x[4];
                float acc = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    thread_x[i] = static_cast<float>(y_shared[t * Dv + lane * 4 + i]);
                    acc += thread_x[i] * thread_x[i];
                }
                acc = simd_sum(acc);
                const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + as_type<float>((uint)EPS_BITS));
                const auto w4 = *reinterpret_cast<const device vec<InT, 4>*>(w + lane * 4);
                const auto z4 = *reinterpret_cast<const device vec<InT, 4>*>(proj + ((b_idx * T + t) * PW + Z_OFF + hv_idx * Dv + lane * 4));
                vec<InT, 4> out4;
                for (int i = 0; i < 4; ++i) {
                    InT normalized = w4[i] * static_cast<InT>(thread_x[i] * inv_mean);
                    const float zg = mlx_sigmoid(static_cast<float>(z4[i]));
                    out4[i] = static_cast<InT>(zg * static_cast<float>(normalized));
                }
                *reinterpret_cast<device vec<InT, 4>*>(gated + (((b_idx * T + t) * Hv + hv_idx) * Dv + lane * 4)) = out4;
            }
        }
        """#
}
