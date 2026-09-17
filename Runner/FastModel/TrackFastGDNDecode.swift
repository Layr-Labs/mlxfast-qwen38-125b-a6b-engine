// Decode GDN preparation, state transition and gated normalization in one launch.
// Each threadgroup owns one value head; each SIMD group owns four value rows.
// Per-row arithmetic, intermediate BF16 conversions and reduction lanes follow
// TrackFastKernels.prepSource, leanSource and gatedRMSSource.

import MLX

enum TrackFastGDNDecode {
    private static let kernel = MLXFast.metalKernel(
        name: "track_gdn_decode_complete",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "state_in", "w"],
        outputNames: ["state_out", "gated", "conv_out"],
        source: source, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func apply(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray, normW: MLXArray,
        zOffset: Int, eps: Float, capture: Bool, geometry g: TrackFastKernels.GDNGeometry
    ) -> (gated: MLXArray, stateOut: MLXArray, convOut: MLXArray)? {
        guard !capture, proj.ndim == 3, proj.dim(1) == 1, proj.dtype == .bfloat16,
            stateIn.dtype == .float32, g.dk == 128, g.dv == 128,
            g.hk > 0, g.hv % g.hk == 0, g.convKernel > 1,
            g.convDim == (2 * g.hk + g.hv) * 128, g.projWidth == proj.dim(2),
            zOffset >= 0, zOffset + g.hv * g.dv <= g.projWidth,
            g.bOffset >= 0, g.bOffset + g.hv <= g.projWidth,
            g.aOffset >= 0, g.aOffset + g.hv <= g.projWidth
        else { return nil }
        let B = proj.dim(0)
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
            ],
            grid: (32, g.dv / 4, B * g.hv), threadGroup: (32, g.dv / 4, 1),
            outputShapes: [[B, g.hv, g.dv, g.dk], [B, 1, g.hv * g.dv], [B, g.convKernel - 1, g.convDim]],
            outputDTypes: [stateIn.dtype, proj.dtype, proj.dtype])
        return (result[1], result[0], result[2])
    }

    private static let source = #"""
        static_assert(Dk == 128 && Dv == 128, "GDN head geometry");
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        constexpr int KM1 = KC - 1;
        threadgroup InT q_shared[Dk];
        threadgroup InT k_shared[Dk];
        threadgroup InT v_shared[Dv];
        threadgroup float gb_shared[2];
        if (sg < 3) {
            const uint vec = sg == 0 ? hk_idx : (sg == 1 ? Hk + hk_idx : 2 * Hk + hv_idx);
            const device InT* proj_b = proj + b_idx * PW;
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
                    cacc += win(j, ch) * conv_w[ch * KC + j];
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
                    if (sg == 0) { q_shared[d] = q_mul * normalized; }
                    else { k_shared[d] = k_mul * normalized; }
                }
            } else {
                for (int i = 0; i < 4; ++i) { v_shared[lane * 4 + i] = static_cast<InT>(thread_x[i]); }
            }
            if (sg == 2 || hv_idx % (Hv / Hk) == 0) {
                device InT* o_conv = conv_out + b_idx * KM1 * CONV_DIM;
                for (int i = 0; i < 4; ++i) {
                    const uint ch = vec * 128 + lane * 4 + i;
                    for (int j = 0; j < KM1; ++j) {
                        o_conv[(uint)(j * CONV_DIM) + ch] = static_cast<InT>(win(1 + j, ch));
                    }
                }
            }
        }
        if (sg == 0 && lane == 0) {
            const device InT* row = proj + b_idx * PW;
            const InT b_raw = row[B_OFF + hv_idx];
            gb_shared[1] = static_cast<float>(mlx_sigmoid(b_raw));
            const InT ax = row[A_OFF + hv_idx] + dt_bias[hv_idx];
            const InT sp = mlx_logaddexp0(ax);
            gb_shared[0] = metal::precise::exp(neg_exp_alog[hv_idx] * sp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup InT* q_ = q_shared;
        const threadgroup InT* k_ = k_shared;
        const threadgroup InT* v_ = v_shared;
        const float gate_decay = gb_shared[0];
        const float gate_beta = gb_shared[1];
        threadgroup InT y_shared[Dv];
        threadgroup float norm_sums[32];
        for (int r = 0; r < RPS; ++r) {
            const uint dv_idx = sg * RPS + r;
            const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
            float state[4];
            constexpr bool vec4 = metal::is_same<StT, float>::value;
            if constexpr (vec4) {
                const float4 s4 = *reinterpret_cast<const device float4*>(i_state + 4 * lane);
                state[0] = s4.x; state[1] = s4.y; state[2] = s4.z; state[3] = s4.w;
            } else {
                for (int i = 0; i < 4; ++i) { state[i] = static_cast<float>(i_state[4 * lane + i]); }
            }
            float kv_mem = 0.0f;
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_compensation = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const int s_idx = 4 * lane + i;
                    state[i] = state[i] * gate_decay;
                    auto product = state[i] * static_cast<float>(k_[s_idx]);
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
                state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
                out += state[i] * static_cast<float>(q_[s_idx]);
            }
            out = simd_sum(out);
            if (lane == 0) {
                const InT value = static_cast<InT>(out);
                y_shared[dv_idx] = value;
            }
            device StT* o_state = state_out + (n * Dv + dv_idx) * Dk;
            if constexpr (vec4) {
                *reinterpret_cast<device float4*>(o_state + 4 * lane) = float4(state[0], state[1], state[2], state[3]);
            } else {
                for (int i = 0; i < 4; ++i) { o_state[4 * lane + i] = static_cast<StT>(state[i]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float thread_x[4];
        if (sg == 0) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                thread_x[i] = static_cast<float>(y_shared[lane * 4 + i]);
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            norm_sums[lane] = lane == 0 ? acc : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const float acc = simd_sum(norm_sums[lane]);
            const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + as_type<float>((uint)EPS_BITS));
            for (int i = 0; i < 4; ++i) {
                const uint d = lane * 4 + i;
                InT normalized = w[d] * static_cast<InT>(thread_x[i] * inv_mean);
                const float z = static_cast<float>(proj[b_idx * PW + Z_OFF + hv_idx * Dv + d]);
                const float zg = mlx_sigmoid(z);
                gated[n * Dv + d] = static_cast<InT>(zg * static_cast<float>(normalized));
            }
        }
        """#
}
