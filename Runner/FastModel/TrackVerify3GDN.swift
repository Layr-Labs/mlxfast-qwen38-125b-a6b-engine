// MLXFAST-V3GDN (valve): the decode GDN megafusion for a TWO-ROW window.
//
// At S = 1 the tree runs the whole deltanet block -- causal depthwise
// convolution, q/k/v preparation, the state transition and the gated RMS norm --
// in ONE launch (`track_gdn_decode_complete`). At S = 2 it falls back to three
// launches per layer (`track_gdn_prep` + `track_gdn_lean_two_row` +
// `track_gated_rms`): 36 launches become 106, and the recurrent state
// ([1, Hv, Dv, Dk] float32) crosses DRAM once more than it needs to.
//
// The deltanet recurrence is SEQUENTIAL in the window anyway, so a two-row
// megafusion is serial-semantic by construction: token 0's update runs first,
// token 1's update runs on its result, and each token's per-element arithmetic,
// bf16 roundings and reduction lanes are the S = 1 kernel's, verbatim. The
// convolution window of token t is rows [t, t + KC) of the concatenation
// `conv_state || proj`, which is exactly the window the serial second step would
// see after the first step rotated the conv tail; the emitted conv tail is rows
// [S, S + KC - 1), i.e. the tail after both tokens. `q`, `k` and `v` depend only
// on `proj` and `conv_state`, never on the recurrent state, so both tokens'
// preparations run before the recurrence and the recurrent state is READ once.
//
// A speculative verify window runs with CAPTURE on -- the engine keeps the
// recurrent state and the convolution tail after EVERY position so a partial
// acceptance can roll back -- so the kernel takes a `CAPTURE` arm that writes
// one state and one conv tail per position, in the same per-slot layout the
// three-launch path writes. That is the arm the depth-1 round actually uses;
// the `!CAPTURE` arm keeps the final state only, and both leave every row's
// arithmetic identical to the serial step's.
//
// Nothing here is shared between the two rows except the state register file and
// the launch: no value computed for row 0 enters row 1's arithmetic other than
// through the recurrence the serial path also carries.

import Foundation
import MLX

enum TrackVerify3GDN {

    /// Kill switch, ON by default (see `TrackVerify2TwoRow.routerTwoRowEnabled`).
    nonisolated(unsafe) static let enabled: Bool =
        !FileManager.default.fileExists(atPath: "/tmp/mlx-v2/no-gdn2row")

    private static let kernel = MLXFast.metalKernel(
        name: "track_gdn_decode_complete_2row",
        inputNames: ["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "state_in", "w"],
        outputNames: ["state_out", "gated", "conv_out"],
        source: source, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    /// Same contract as `TrackFastGDNDecode.apply`, for `proj.dim(1) == S <= KC - 1`.
    static func apply(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray, normW: MLXArray,
        zOffset: Int, eps: Float, capture: Bool, geometry g: TrackFastKernels.GDNGeometry
    ) -> (gated: MLXArray, stateOut: MLXArray, convOut: MLXArray)? {
        guard enabled else { return nil }
        return run(
            proj: proj, convState: convState, convW: convW, negExpALog: negExpALog,
            dtBias: dtBias, stateIn: stateIn, normW: normW, zOffset: zOffset, eps: eps,
            capture: capture, geometry: g)
    }

    /// The valve-free entry point (the bit-comparison harness calls this).
    static func run(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray, normW: MLXArray,
        zOffset: Int, eps: Float, capture: Bool, geometry g: TrackFastKernels.GDNGeometry
    ) -> (gated: MLXArray, stateOut: MLXArray, convOut: MLXArray)? {
        guard proj.ndim == 3, proj.dim(1) == 2, proj.dtype == .bfloat16,
            stateIn.dtype == .float32, g.dk == 128, g.dv == 128,
            g.hk > 0, g.hv % g.hk == 0, g.convKernel > 2,
            g.convDim == (2 * g.hk + g.hv) * 128, g.projWidth == proj.dim(2),
            zOffset >= 0, zOffset + g.hv * g.dv <= g.projWidth,
            g.bOffset >= 0, g.bOffset + g.hv <= g.projWidth,
            g.aOffset >= 0, g.aOffset + g.hv <= g.projWidth
        else { return nil }
        let B = proj.dim(0), S = proj.dim(1)
        let slots = capture ? B * S : B
        guard B == 1, S <= g.convKernel - 1,
            convState.shape == [B, g.convKernel - 1, g.convDim],
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
                ("Z_OFF", zOffset), ("EPS_BITS", Int(eps.bitPattern)), ("RPS", 4), ("S", S),
                ("CAPTURE", capture),
            ],
            grid: (32, g.dv / 4, B * g.hv), threadGroup: (32, g.dv / 4, 1),
            outputShapes: [
                [slots, g.hv, g.dv, g.dk], [B, S, g.hv * g.dv],
                [slots, g.convKernel - 1, g.convDim],
            ],
            outputDTypes: [stateIn.dtype, proj.dtype, proj.dtype])
        return (result[1], result[0], result[2])
    }

    private static let source = #"""
        static_assert(Dk == 128 && Dv == 128, "GDN head geometry");
        static_assert(S >= 1 && S <= KC - 1, "two-row megafusion needs S <= KC - 1");
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        constexpr int KM1 = KC - 1;
        threadgroup InT q_shared[S][Dk];
        threadgroup InT k_shared[S][Dk];
        threadgroup InT v_shared[S][Dv];
        threadgroup float gb_shared[S][2];
        if (sg < 3) {
            const uint vec = sg == 0 ? hk_idx : (sg == 1 ? Hk + hk_idx : 2 * Hk + hv_idx);
            const device InT* proj_b = proj + (size_t)b_idx * (size_t)S * (size_t)PW;
            const device InT* cst_b = conv_state + b_idx * KM1 * CONV_DIM;
            // Row r of the concatenation `conv_state || proj` for this window.
            auto win = [&](int r, uint ch) -> float {
                if (r < KM1) { return static_cast<float>(cst_b[(uint)(r * CONV_DIM) + ch]); }
                return static_cast<float>(proj_b[(size_t)(r - KM1) * (size_t)PW + ch]);
            };
            for (int t = 0; t < S; ++t) {
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
                        if (sg == 0) { q_shared[t][d] = q_mul * normalized; }
                        else { k_shared[t][d] = k_mul * normalized; }
                    }
                } else {
                    for (int i = 0; i < 4; ++i) { v_shared[t][lane * 4 + i] = static_cast<InT>(thread_x[i]); }
                }
            }
            // The conv tail. Without capture only the tail after the whole
            // window is kept (rows [S, S + KM1)); with capture the tail after
            // EVERY position is kept, in the same per-slot layout the three-launch
            // path writes (slot b*S + t holds rows [t+1, t+1+KM1)).
            if (sg == 2 || hv_idx % (Hv / Hk) == 0) {
                for (int t = CAPTURE ? 0 : (S - 1); t < S; ++t) {
                    const uint cslot = CAPTURE ? (uint)(b_idx * S + t) : b_idx;
                    device InT* o_conv = conv_out + cslot * KM1 * CONV_DIM;
                    for (int i = 0; i < 4; ++i) {
                        const uint ch = vec * 128 + lane * 4 + i;
                        for (int j = 0; j < KM1; ++j) {
                            o_conv[(uint)(j * CONV_DIM) + ch] =
                                static_cast<InT>(win(t + 1 + j, ch));
                        }
                    }
                }
            }
        }
        if (sg == 0 && lane < (uint)S) {
            const device InT* row = proj + ((size_t)b_idx * (size_t)S + (size_t)lane) * (size_t)PW;
            const InT b_raw = row[B_OFF + hv_idx];
            gb_shared[lane][1] = static_cast<float>(mlx_sigmoid(b_raw));
            const InT ax = row[A_OFF + hv_idx] + dt_bias[hv_idx];
            const InT sp = mlx_logaddexp0(ax);
            gb_shared[lane][0] = metal::precise::exp(neg_exp_alog[hv_idx] * sp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup InT y_shared[S][Dv];
        threadgroup float norm_sums[S][32];
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
            // The window's tokens, in order: token t's transition is the serial
            // step's transition applied to the state token t - 1 left behind.
            for (int t = 0; t < S; ++t) {
                const threadgroup InT* q_ = q_shared[t];
                const threadgroup InT* k_ = k_shared[t];
                const threadgroup InT* v_ = v_shared[t];
                const float gate_decay = gb_shared[t][0];
                const float gate_beta = gb_shared[t][1];
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
                if (lane == 0) { y_shared[t][dv_idx] = static_cast<InT>(out); }
                if (CAPTURE) {
                    const uint sslot = (uint)(b_idx * S + t);
                    device StT* o_cap =
                        state_out + (((size_t)sslot * Hv + hv_idx) * Dv + dv_idx) * Dk;
                    if constexpr (vec4) {
                        *reinterpret_cast<device float4*>(o_cap + 4 * lane) =
                            float4(state[0], state[1], state[2], state[3]);
                    } else {
                        for (int i = 0; i < 4; ++i) { o_cap[4 * lane + i] = static_cast<StT>(state[i]); }
                    }
                }
            }
            if (!CAPTURE) {
                device StT* o_state = state_out + (n * Dv + dv_idx) * Dk;
                if constexpr (vec4) {
                    *reinterpret_cast<device float4*>(o_state + 4 * lane) = float4(state[0], state[1], state[2], state[3]);
                } else {
                    for (int i = 0; i < 4; ++i) { o_state[4 * lane + i] = static_cast<StT>(state[i]); }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float thread_x[S][4];
        if (sg == 0) {
            for (int t = 0; t < S; ++t) {
                float acc = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    thread_x[t][i] = static_cast<float>(y_shared[t][lane * 4 + i]);
                    acc += thread_x[t][i] * thread_x[t][i];
                }
                acc = simd_sum(acc);
                norm_sums[t][lane] = lane == 0 ? acc : 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (int t = 0; t < S; ++t) {
                const float acc = simd_sum(norm_sums[t][lane]);
                const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + as_type<float>((uint)EPS_BITS));
                const device InT* prow = proj + ((size_t)b_idx * (size_t)S + (size_t)t) * (size_t)PW;
                device InT* grow = gated + ((size_t)b_idx * (size_t)S + (size_t)t) * (size_t)(Hv * Dv);
                for (int i = 0; i < 4; ++i) {
                    const uint d = lane * 4 + i;
                    InT normalized = w[d] * static_cast<InT>(thread_x[t][i] * inv_mean);
                    const float z = static_cast<float>(prow[Z_OFF + hv_idx * Dv + d]);
                    const float zg = mlx_sigmoid(z);
                    grow[hv_idx * Dv + d] = static_cast<InT>(zg * static_cast<float>(normalized));
                }
            }
        }
        """#
}
