// TrackFastKernels.swift -- fused Metal kernels for the track's fast forward pass.
//
// ONE launch per gated-deltanet layer replaces the engine's chain of
// concat + depthwise conv + silu + split + two RMS norms + the nine-op gate
// chain + the recurrence launch. The recurrence arithmetic (Kahan-compensated
// state update, f32 state) is the fork's `gated_delta_step` kernel, kept
// verbatim so the per-position numerics match the reference path; what is
// new is that the causal convolution, the q/k norms and the decay/beta gates
// are computed in-kernel from the fused projection output, and that a
// capture-verify window writes the state after EVERY position in the same
// launch (`CAPTURE`), instead of one launch per position.

import Foundation
import MLX

enum TrackFastKernels {

    /// Fused GDN step. See the file note.
    ///
    /// Inputs (all row-contiguous):
    ///   proj       [B, T, PROJ_W] InT   fused in_proj output: qkv | z | b | a
    ///   conv_state [B, KC-1, CONV_DIM] InT
    ///   conv_w     [CONV_DIM, KC] InT
    ///   neg_exp_alog [Hv] float32  (-exp(A_log))
    ///   dt_bias    [Hv] InT
    ///   state_in   [B, Hv, Dv, Dk] StT
    ///   T          scalar int32
    /// Outputs:
    ///   y          [B, T, Hv, Dv] InT   (pre output-norm recurrence output)
    ///   conv_out   [B * (CAPTURE ? T : 1), KC-1, CONV_DIM] InT
    ///   state_out  [B * (CAPTURE ? T : 1), Hv, Dv, Dk] StT
    static let gdnSource = """
        constexpr int n_per_t = Dk / 32;
        constexpr int KM1 = KC - 1;
        constexpr int q_off = 0;
        constexpr int k_off = Dk * Hk;
        constexpr int v_off = 2 * Dk * Hk;
        constexpr int z_dummy = 0;
        (void)z_dummy;

        const uint n = thread_position_in_grid.z;          // b * Hv + hv
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint dk_lane = thread_position_in_threadgroup.x; // 0..31
        const uint dv_idx = thread_position_in_grid.y;         // 0..Dv-1
        const int T_ = T;

        const device InT* proj_b = proj + (uint)(b_idx * T_ * PROJ_W);
        const device InT* cst_b = conv_state + (uint)(b_idx * KM1 * CONV_DIM);

        // Window row r (0 <= r < T + KM1): r < KM1 -> conv_state[r], else proj[r - KM1].
        auto win = [&](int r, int ch) -> float {
            if (r < KM1) {
                return static_cast<float>(cst_b[(uint)(r * CONV_DIM + ch)]);
            }
            return static_cast<float>(proj_b[(uint)((r - KM1) * PROJ_W + ch)]);
        };
        // Causal depthwise conv at position t for channel ch, then silu, with
        // the same bf16 rounding points as conv1d -> silu on arrays.
        auto conv_silu = [&](int t, int ch) -> float {
            float acc = 0.0f;
            for (int j = 0; j < KC; ++j) {
                acc += win(t + j, ch) * static_cast<float>(conv_w[(uint)(ch * KC + j)]);
            }
            float c0 = static_cast<float>(static_cast<InT>(acc));
            float s = static_cast<float>(static_cast<InT>(1.0f / (1.0f + metal::exp(-c0))));
            return static_cast<float>(static_cast<InT>(c0 * s));
        };

        // Per-thread recurrent state slice: dv row `dv_idx`, dk lanes.
        float state[n_per_t];
        {
            const device StT* i_state = state_in + (uint)((n * Dv + dv_idx) * Dk);
            for (int i = 0; i < n_per_t; ++i) {
                state[i] = static_cast<float>(i_state[n_per_t * dk_lane + i]);
            }
        }

        const float inv_scale = metal::rsqrt(static_cast<float>(Dk));
        const float q_mul = static_cast<float>(static_cast<InT>(inv_scale * inv_scale));
        const float k_mul = static_cast<float>(static_cast<InT>(inv_scale));
        const float neg_ea = neg_exp_alog[hv_idx];
        const float dtb = static_cast<float>(dt_bias[hv_idx]);

        device InT* y_b = y + (uint)(b_idx * T_ * Hv * Dv + hv_idx * Dv);

        for (int t = 0; t < T_; ++t) {
            // --- q / k for this key head: conv + silu, then RMS norm + scale.
            float qv[n_per_t];
            float kv[n_per_t];
            float qss = 0.0f;
            float kss = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                const int d = n_per_t * dk_lane + i;
                qv[i] = conv_silu(t, q_off + hk_idx * Dk + d);
                kv[i] = conv_silu(t, k_off + hk_idx * Dk + d);
                qss += qv[i] * qv[i];
                kss += kv[i] * kv[i];
            }
            qss = simd_sum(qss);
            kss = simd_sum(kss);
            const float qn = metal::rsqrt(qss / static_cast<float>(Dk) + 1e-6f);
            const float kn = metal::rsqrt(kss / static_cast<float>(Dk) + 1e-6f);
            for (int i = 0; i < n_per_t; ++i) {
                float qq = static_cast<float>(static_cast<InT>(qv[i] * qn));
                qv[i] = static_cast<float>(static_cast<InT>(qq * q_mul));
                float kk = static_cast<float>(static_cast<InT>(kv[i] * kn));
                kv[i] = static_cast<float>(static_cast<InT>(kk * k_mul));
            }
            // --- v for this dv row.
            const float vv = conv_silu(t, v_off + hv_idx * Dv + dv_idx);
            // --- gates: beta = sigmoid(b) (bf16), g = exp(-exp(A_log) * softplus(a + dt_bias)).
            const device InT* row = proj_b + (uint)(t * PROJ_W);
            const float b_raw = static_cast<float>(row[B_OFF + hv_idx]);
            const float a_raw = static_cast<float>(row[A_OFF + hv_idx]);
            const float beta = static_cast<float>(static_cast<InT>(1.0f / (1.0f + metal::exp(-b_raw))));
            const float ax = static_cast<float>(static_cast<InT>(a_raw + dtb));
            const float sp_f = metal::max(ax, 0.0f) + log1p(metal::exp(-metal::abs(ax)));
            const float sp = static_cast<float>(static_cast<InT>(sp_f));
            const float g = metal::exp(neg_ea * sp);

            // --- recurrence (fork kernel body, verbatim arithmetic).
            float kv_mem = 0.0f;
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_compensation = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    state[i] = state[i] * g;
                    auto product = state[i] * kv[i];
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem + corrected;
                    kv_compensation = (next_sum - kv_mem) - corrected;
                    kv_mem = next_sum;
                }
            }
            kv_mem = simd_sum(kv_mem);
            const float delta = (vv - kv_mem) * beta;
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                state[i] = state[i] + kv[i] * delta;
                out += state[i] * qv[i];
            }
            out = simd_sum(out);
            if (dk_lane == 0) {
                y_b[(uint)(t * Hv * Dv + dv_idx)] = static_cast<InT>(out);
            }

            const bool emit = CAPTURE || (t == T_ - 1);
            if (emit) {
                const uint slot = CAPTURE ? (uint)(b_idx * T_ + t) : (uint)b_idx;
                // recurrent state after position t
                device StT* o_state = state_out + (uint)(((slot * Hv + hv_idx) * Dv + dv_idx) * Dk);
                for (int i = 0; i < n_per_t; ++i) {
                    o_state[n_per_t * dk_lane + i] = static_cast<StT>(state[i]);
                }
                // conv tail after position t: window rows t+1 .. t+KM1
                device InT* o_conv = conv_out + (uint)(slot * KM1 * CONV_DIM);
                if (thread_position_in_threadgroup.y == 0) {
                    for (int j = 0; j < KM1; ++j) {
                        for (int i = 0; i < n_per_t; ++i) {
                            const int d = n_per_t * dk_lane + i;
                            const int qc = q_off + hk_idx * Dk + d;
                            const int kc = k_off + hk_idx * Dk + d;
                            o_conv[(uint)(j * CONV_DIM + qc)] = static_cast<InT>(win(t + 1 + j, qc));
                            o_conv[(uint)(j * CONV_DIM + kc)] = static_cast<InT>(win(t + 1 + j, kc));
                        }
                    }
                }
                if (dk_lane == 0) {
                    const int vc = v_off + hv_idx * Dv + dv_idx;
                    for (int j = 0; j < KM1; ++j) {
                        o_conv[(uint)(j * CONV_DIM + vc)] = static_cast<InT>(win(t + 1 + j, vc));
                    }
                }
            }
        }
        """

    nonisolated(unsafe) static let gdnKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "track_gdn_fused",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "state_in", "T"],
        outputNames: ["y", "conv_out", "state_out"],
        source: gdnSource,
        ensureRowContiguous: false)

    struct GDNGeometry {
        let projWidth: Int      // PROJ_W
        let convDim: Int        // CONV_DIM = 2*Hk*Dk + Hv*Dv
        let convKernel: Int     // KC
        let hk: Int, hv: Int, dk: Int, dv: Int
        let bOffset: Int        // B_OFF
        let aOffset: Int        // A_OFF
    }

    /// Run the fused GDN kernel. Returns (y [B,T,Hv,Dv], convOut, stateOut).
    static func gdn(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray,
        T: Int, capture: Bool, geometry g: GDNGeometry
    ) -> (y: MLXArray, convOut: MLXArray, stateOut: MLXArray) {
        let B = proj.dim(0)
        let slots = capture ? B * T : B
        let outs = gdnKernel(
            [proj, convState, convW, negExpALog, dtBias, stateIn, MLXArray(Int32(T))],
            template: [
                ("InT", proj.dtype), ("StT", stateIn.dtype),
                ("Dk", g.dk), ("Dv", g.dv), ("Hk", g.hk), ("Hv", g.hv),
                ("KC", g.convKernel), ("PROJ_W", g.projWidth), ("CONV_DIM", g.convDim),
                ("B_OFF", g.bOffset), ("A_OFF", g.aOffset), ("CAPTURE", capture),
            ],
            grid: (32, g.dv, B * g.hv),
            threadGroup: (32, 4, 1),
            outputShapes: [
                [B, T, g.hv, g.dv],
                [slots, g.convKernel - 1, g.convDim],
                [slots, g.hv, g.dv, g.dk],
            ],
            outputDTypes: [proj.dtype, proj.dtype, stateIn.dtype])
        return (outs[0], outs[1], outs[2])
    }
}
