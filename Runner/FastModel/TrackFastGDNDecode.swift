// Per-row arithmetic, intermediate BF16 conversions and reduction lanes follow
// TrackFastKernels.prepSource, leanSource and gatedRMSSource.

import MLX

enum TrackFastGDNDecode {
    private static let kernel = MLXFast.metalKernel(
        name: "track_gdn_decode_complete",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "state_in", "w", "journal_in"],
        outputNames: ["state_out", "gated", "conv_out", "journal_out"],
        source: source, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func apply(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray, normW: MLXArray,
        pendingJournal: MLXArray?, zOffset: Int, eps: Float, capture: Bool,
        geometry g: TrackFastKernels.GDNGeometry
    ) -> (gated: MLXArray, stateOut: MLXArray, convOut: MLXArray, journal: MLXArray?)? {
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
        let journalStride = g.hv * g.dk + 2 * g.hv * g.dv + 2 * g.hv
        let hasJournal = pendingJournal?.shape == [B, journalStride]
        let journalIn = pendingJournal ?? convState
        let result = kernel(
            [proj, convState, convW, negExpALog, dtBias, stateIn, normW, journalIn],
            template: [
                ("InT", proj.dtype), ("StT", stateIn.dtype), ("Dk", g.dk), ("Dv", g.dv),
                ("Hk", g.hk), ("Hv", g.hv), ("KC", g.convKernel), ("CONV_DIM", g.convDim),
                ("PW", g.projWidth), ("B_OFF", g.bOffset), ("A_OFF", g.aOffset),
                ("Z_OFF", zOffset), ("EPS_BITS", Int(eps.bitPattern)), ("RPS", 4),
                ("J_KEY_OFF", 0), ("J_DELTA_OFF", g.hv * g.dk),
                ("J_DECAY_OFF", g.hv * g.dk + 2 * g.hv * g.dv),
                ("J_STRIDE", journalStride), ("HAS_JOURNAL", hasJournal),
            ],
            grid: (32, g.dv / 4, B * g.hv), threadGroup: (32, g.dv / 4, 1),
            outputShapes: [
                hasJournal ? [B, g.hv, g.dv, g.dk] : [1],
                [B, 1, g.hv * g.dv], [B, g.convKernel - 1, g.convDim],
                hasJournal ? [1] : [B, journalStride],
            ],
            outputDTypes: [stateIn.dtype, proj.dtype, proj.dtype, proj.dtype])
        return (result[1], hasJournal ? result[0] : stateIn, result[2], hasJournal ? nil : result[3])
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
        static_assert(RPS % 2 == 0, "paired GDN rows require an even row count");
        float4 next_state[2];
        if constexpr (metal::is_same<StT, float>::value) {
            const device StT* first_state = state_in + (n * Dv + sg * RPS) * Dk;
            for (int p = 0; p < 2; ++p) {
                next_state[p] = *reinterpret_cast<const device float4*>(first_state + p * Dk + 4 * lane);
            }
        }
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
        const device InT* journal = journal_in + b_idx * J_STRIDE;
        device InT* next_journal = journal_out + b_idx * J_STRIDE;
        auto journal_float = [&](int off) -> float {
            const uint lo = (uint)as_type<ushort>(journal[off]);
            const uint hi = (uint)as_type<ushort>(journal[off + 1]);
            return as_type<float>(lo | (hi << 16));
        };
        auto store_journal_float = [&](int off, float value) {
            const uint bits = as_type<uint>(value);
            next_journal[off] = as_type<InT>((ushort)(bits & 0xffffu));
            next_journal[off + 1] = as_type<InT>((ushort)(bits >> 16));
        };
        const float4 local_q = float4(
            static_cast<float>(q_[4 * lane]), static_cast<float>(q_[4 * lane + 1]),
            static_cast<float>(q_[4 * lane + 2]), static_cast<float>(q_[4 * lane + 3]));
        const float4 local_k = float4(
            static_cast<float>(k_[4 * lane]), static_cast<float>(k_[4 * lane + 1]),
            static_cast<float>(k_[4 * lane + 2]), static_cast<float>(k_[4 * lane + 3]));
        // A journal's key and decay are invariant across this simdgroup's rows.
        float pending_decay = 0.0f;
        float4 pending_key = float4(0.0f);
        if constexpr (HAS_JOURNAL) {
            pending_decay = journal_float(J_DECAY_OFF + 2 * hv_idx);
            for (int i = 0; i < 4; ++i) {
                pending_key[i] = static_cast<float>(journal[J_KEY_OFF + hv_idx * Dk + 4 * lane + i]);
            }
        }
        threadgroup InT y_shared[Dv];
        for (int r = 0; r < RPS; r += 2) {
            float state[2][4];
            // Load both independent rows and prefetch the next pair. No row's
            // arithmetic is reassociated; only work on distinct rows overlaps.
            for (int p = 0; p < 2; ++p) {
                const uint dv_idx = sg * RPS + r + p;
                const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
                if constexpr (metal::is_same<StT, float>::value) {
                    const float4 s4 = next_state[p];
                    if (r + 2 < RPS) {
                        next_state[p] = *reinterpret_cast<const device float4*>(i_state + 2 * Dk + 4 * lane);
                    }
                    state[p][0] = s4.x; state[p][1] = s4.y; state[p][2] = s4.z; state[p][3] = s4.w;
                } else {
                    for (int i = 0; i < 4; ++i) { state[p][i] = static_cast<float>(i_state[4 * lane + i]); }
                }
                if constexpr (HAS_JOURNAL) {
                    const float pending_delta = journal_float(J_DELTA_OFF + 2 * (hv_idx * Dv + dv_idx));
                    for (int i = 0; i < 4; ++i) {
                        state[p][i] = state[p][i] * pending_decay;
                        state[p][i] = state[p][i] + pending_key[i] * pending_delta;
                    }
                }
            }
            float kv_mem[2];
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                for (int p = 0; p < 2; ++p) {
                    state[p][0] = state[p][0] * gate_decay;
                    kv_mem[p] = 0.0f + state[p][0] * local_k[0];
                    float kv_compensation = 0.0f;
                    for (int i = 1; i < 4; ++i) {
                        state[p][i] = state[p][i] * gate_decay;
                        auto product = state[p][i] * local_k[i];
                        auto corrected = product - kv_compensation;
                        auto next_sum = kv_mem[p] + corrected;
                        if (i + 1 < 4) { kv_compensation = (next_sum - kv_mem[p]) - corrected; }
                        kv_mem[p] = next_sum;
                    }
                }
            }
            for (int p = 0; p < 2; ++p) { kv_mem[p] = simd_sum(kv_mem[p]); }
            float out[2];
            for (int p = 0; p < 2; ++p) {
                const uint dv_idx = sg * RPS + r + p;
                const float delta = (static_cast<float>(v_[dv_idx]) - kv_mem[p]) * gate_beta;
                if constexpr (!HAS_JOURNAL) {
                    if (lane == 0) {
                        store_journal_float(J_DELTA_OFF + 2 * (hv_idx * Dv + dv_idx), delta);
                    }
                }
                out[p] = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    state[p][i] = state[p][i] + local_k[i] * delta;
                    out[p] += state[p][i] * local_q[i];
                }
            }
            for (int p = 0; p < 2; ++p) { out[p] = simd_sum(out[p]); }
            for (int p = 0; p < 2; ++p) {
                const uint dv_idx = sg * RPS + r + p;
                if (lane == 0) { y_shared[dv_idx] = static_cast<InT>(out[p]); }
                if constexpr (HAS_JOURNAL) {
                    device StT* o_state = state_out + (n * Dv + dv_idx) * Dk;
                    if constexpr (metal::is_same<StT, float>::value) {
                        *reinterpret_cast<device float4*>(o_state + 4 * lane) = float4(state[p][0], state[p][1], state[p][2], state[p][3]);
                    } else {
                        for (int i = 0; i < 4; ++i) { o_state[4 * lane + i] = static_cast<StT>(state[p][i]); }
                    }
                }
            }
        }
        if constexpr (!HAS_JOURNAL) {
            if (sg == 0) {
                for (int i = 0; i < 4; ++i) {
                    next_journal[J_KEY_OFF + hv_idx * Dk + 4 * lane + i] = k_[4 * lane + i];
                }
            }
            if (sg == 0 && lane == 0) {
                store_journal_float(J_DECAY_OFF + 2 * hv_idx, gate_decay);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float thread_x[4];
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                thread_x[i] = static_cast<float>(y_shared[lane * 4 + i]);
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + as_type<float>((uint)EPS_BITS));
            const auto w4 = *reinterpret_cast<const device vec<InT, 4>*>(w + lane * 4);
            const auto z4 = *reinterpret_cast<const device vec<InT, 4>*>(proj + (b_idx * PW + Z_OFF + hv_idx * Dv + lane * 4));
            vec<InT, 4> out4;
            for (int i = 0; i < 4; ++i) {
                InT normalized = w4[i] * static_cast<InT>(thread_x[i] * inv_mean);
                const float zg = mlx_sigmoid(static_cast<float>(z4[i]));
                out4[i] = static_cast<InT>(zg * static_cast<float>(normalized));
            }
            *reinterpret_cast<device vec<InT, 4>*>(gated + (n * Dv + lane * 4)) = out4;
        }
        """#
}
