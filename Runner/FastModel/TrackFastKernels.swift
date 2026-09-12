// TrackFastKernels.swift -- the gated-deltanet kernels of the fast forward pass.
//
// Two launches per deltanet layer replace the engine's chain of ~20:
//
//   * `track_gdn_prep` runs the causal depthwise convolution + silu, the q/k
//     RMS norms with their scales, the decay/beta gates, and writes the conv
//     tails the recurrent state keeps -- one thread per channel per position,
//     fully parallel over the window.
//   * `track_gdn_lean` is the fork's delta-rule recurrence (Kahan-compensated
//     f32 state, verbatim arithmetic) over the prepared inputs, extended so a
//     capture-verify window writes the state after EVERY position in the same
//     launch (`CAPTURE`), instead of one launch per position.
//
// Rounding follows the engine's op chain: wherever it materialises a bf16
// array, the kernels round through InT at the same point.

import Foundation
import MLX

enum TrackFastKernels {

    struct GDNGeometry {
        let projWidth: Int  // PROJ_W
        let convDim: Int  // CONV_DIM = 2*Hk*Dk + Hv*Dv
        let convKernel: Int  // KC
        let hk: Int, hv: Int, dk: Int, dv: Int
        let bOffset: Int  // B_OFF
        let aOffset: Int  // A_OFF
    }

    // MARK: prep (bit-exact with conv1d -> compiled silu -> rmsNorm(none) * scale, sigmoid, logAddExp)

    /// grid (32, CONV_DIM/128, B*T), threadgroup (32, 4, 1): one simdgroup per
    /// 128-channel vector, four consecutive channels per lane (the layout of
    /// `rms_single_row` at axis 128). Vector index: [0, Hk) q heads, [Hk, 2Hk)
    /// k heads, then the Hv value heads.
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
        """

    nonisolated(unsafe) static let prepKernel = MLXFast.metalKernel(
        name: "track_gdn_prep",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias"],
        outputNames: ["qn", "kn", "vv", "g", "beta", "conv_out"],
        source: prepSource, header: exactHeader, ensureRowContiguous: true)

    /// The prep half alone (tests).
    static func gdnPrep(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, T: Int, capture: Bool, geometry g: GDNGeometry
    ) -> [MLXArray] {
        let B = proj.dim(0)
        let slots = capture ? B * T : B
        precondition(g.dk == 128 && g.dv == 128 && g.convDim % 128 == 0)
        return prepKernel(
            [proj, convState, convW, negExpALog, dtBias],
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

    // MARK: lean recurrence

    static let leanSource = """
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
        const uint dv_idx = thread_position_in_grid.y;
        const device float* g_ = g + b_idx * T_ * Hv;
        const device float* beta_ = beta + b_idx * T_ * Hv;
        const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }
        for (int t = 0; t < T_; ++t) {
            float kv_mem = 0.0f;
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_compensation = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    const int s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    auto product = state[i] * static_cast<float>(k_[s_idx]);
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem + corrected;
                    kv_compensation = (next_sum - kv_mem) - corrected;
                    kv_mem = next_sum;
                }
            }
            kv_mem = simd_sum(kv_mem);
            const float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                const int s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
                out += state[i] * static_cast<float>(q_[s_idx]);
            }
            out = simd_sum(out);
            if (dk_idx == 0) { y_[dv_idx] = static_cast<InT>(out); }
            if (CAPTURE || t == T_ - 1) {
                const uint slot = CAPTURE ? (b_idx * T_ + t) : b_idx;
                device StT* o_state = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;
                for (int i = 0; i < n_per_t; ++i) {
                    o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
                }
            }
            q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        """

    nonisolated(unsafe) static let leanKernel = MLXFast.metalKernel(
        name: "track_gdn_lean",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: leanSource, ensureRowContiguous: true)

    /// The whole deltanet core: prep + recurrence. Returns y [B,T,Hv,Dv],
    /// the conv tails and the recurrent state (per position when `capture`).
    static func gdn(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray,
        T: Int, capture: Bool, geometry g: GDNGeometry
    ) -> (y: MLXArray, convOut: MLXArray, stateOut: MLXArray) {
        let B = proj.dim(0)
        let slots = capture ? B * T : B
        precondition(g.dk == 128 && g.dv == 128 && g.convDim % 128 == 0)
        let prep = gdnPrep(
            proj: proj, convState: convState, convW: convW, negExpALog: negExpALog,
            dtBias: dtBias, T: T, capture: capture, geometry: g)
        let rec = leanKernel(
            [prep[0], prep[1], prep[2], prep[3], prep[4], stateIn, MLXArray(Int32(T))],
            template: [
                ("InT", proj.dtype), ("StT", stateIn.dtype), ("Dk", g.dk), ("Dv", g.dv),
                ("Hk", g.hk), ("Hv", g.hv), ("CAPTURE", capture),
            ],
            grid: (32, g.dv, B * g.hv), threadGroup: (32, 4, 1),
            outputShapes: [[B, T, g.hv, g.dv], [slots, g.hv, g.dv, g.dk]],
            outputDTypes: [proj.dtype, stateIn.dtype])
        return (rec[0], prep[5], rec[1])
    }
}
