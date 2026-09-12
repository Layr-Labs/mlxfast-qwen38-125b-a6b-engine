// Decode GDN preparation, state transition and gated normalization in one launch.
// Each threadgroup owns one value head; each SIMD group interleaves two pairs of value rows.
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
        static_assert(Dk == 128 && Dv == 128 && RPS % 2 == 0, "GDN paired head geometry");
        static_assert(metal::is_same<StT, float>::value, "FP32 GDN state");
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
        for (int r = 0; r < RPS; r += 2) {
            const uint dv_idx = sg * RPS + r;
            const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
            const float4 s0 = *reinterpret_cast<const device float4*>(i_state + 4 * lane);
            const float4 s1 = *reinterpret_cast<const device float4*>(i_state + Dk + 4 * lane);
            float state0[4] = {s0.x, s0.y, s0.z, s0.w};
            float state1[4] = {s1.x, s1.y, s1.z, s1.w};
            float kv_mem0 = 0.0f, kv_mem1 = 0.0f;
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float compensation0 = 0.0f, compensation1 = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const float key = static_cast<float>(k_[4 * lane + i]);
                    state0[i] = state0[i] * gate_decay;
                    auto product0 = state0[i] * key;
                    auto corrected0 = product0 - compensation0;
                    auto sum0 = kv_mem0 + corrected0;
                    compensation0 = (sum0 - kv_mem0) - corrected0;
                    kv_mem0 = sum0;
                    state1[i] = state1[i] * gate_decay;
                    auto product1 = state1[i] * key;
                    auto corrected1 = product1 - compensation1;
                    auto sum1 = kv_mem1 + corrected1;
                    compensation1 = (sum1 - kv_mem1) - corrected1;
                    kv_mem1 = sum1;
                }
            }
            kv_mem0 = simd_sum(kv_mem0);
            kv_mem1 = simd_sum(kv_mem1);
            const float delta0 = (static_cast<float>(v_[dv_idx]) - kv_mem0) * gate_beta;
            const float delta1 = (static_cast<float>(v_[dv_idx + 1]) - kv_mem1) * gate_beta;
            float out0 = 0.0f, out1 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const float key = static_cast<float>(k_[4 * lane + i]);
                const float query = static_cast<float>(q_[4 * lane + i]);
                state0[i] = state0[i] + key * delta0;
                state1[i] = state1[i] + key * delta1;
                out0 += state0[i] * query;
                out1 += state1[i] * query;
            }
            out0 = simd_sum(out0);
            out1 = simd_sum(out1);
            if (lane == 0) {
                y_shared[dv_idx] = static_cast<InT>(out0);
                y_shared[dv_idx + 1] = static_cast<InT>(out1);
            }
            device StT* o_state = state_out + (n * Dv + dv_idx) * Dk;
            *reinterpret_cast<device float4*>(o_state + 4 * lane) = float4(state0[0], state0[1], state0[2], state0[3]);
            *reinterpret_cast<device float4*>(o_state + Dk + 4 * lane) = float4(state1[0], state1[1], state1[2], state1[3]);
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
