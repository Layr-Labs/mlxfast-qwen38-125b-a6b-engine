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
    /// A typed host scalar made once per (value, dtype): a fresh `MLXArray`
    /// per call is a host allocation and a cast launch.
    private static let scalarLock = NSLock()
    nonisolated(unsafe) private static var typedScalars: [String: MLXArray] = [:]
    static func scalar(_ v: Float, dtype: DType) -> MLXArray {
        scalarLock.lock(); defer { scalarLock.unlock() }
        let key = "\(v.bitPattern):\(dtype)"
        if let a = typedScalars[key] { return a }
        let a = MLXArray(v, dtype: dtype); eval(a); typedScalars[key] = a; return a
    }

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
        // Each lane owns n_per_t consecutive state entries: one vector load /
        // store per lane when that is a float4 (same values, one instruction
        // instead of four strided ones; this kernel is load/store bound).
        constexpr bool vec4 = (n_per_t == 4) && metal::is_same<StT, float>::value;
        if constexpr (vec4) {
            const float4 s4 = *reinterpret_cast<const device float4*>(i_state + 4 * dk_idx);
            state[0] = s4.x; state[1] = s4.y; state[2] = s4.z; state[3] = s4.w;
        } else {
            for (int i = 0; i < n_per_t; ++i) {
                state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
            }
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
                if constexpr (vec4) {
                    *reinterpret_cast<device float4*>(o_state + 4 * dk_idx) = float4(state[0], state[1], state[2], state[3]);
                } else {
                    for (int i = 0; i < n_per_t; ++i) {
                        o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
                    }
                }
            }
            q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        """

    nonisolated(unsafe) static let leanKernel = MLXFast.metalKernel(
        name: "track_gdn_lean",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out"],
        source: leanSource, ensureRowContiguous: true)

    // MLXFAST-GDNTILE: Two adjacent value rows share Q/K and gate loads in prefill.
    // Keep the original grid/threadgroup; surplus SIMD groups return uniformly.
    // Per lane: four extra FP32 state values, plus a second sum/compensation or
    // delta/output pair. Actual register allocation and spills require profiling.
    static let leanTwoRowSource = """
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
                y_[dv_idx] = static_cast<InT>(out0);
                y_[dv_idx + 1] = static_cast<InT>(out1);
            }
            if (CAPTURE || t == T_ - 1) {
                const uint slot = CAPTURE ? (b_idx * T_ + t) : b_idx;
                device StT* o_state = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;
                for (int i = 0; i < n_per_t; ++i) {
                    o_state[n_per_t * dk_idx + i] = static_cast<StT>(state0[i]);
                    o_state[Dk + n_per_t * dk_idx + i] = static_cast<StT>(state1[i]);
                }
            }
            q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        """

    // MLXFAST-GDNTILE: Keep the one-row kernel intact for decode and comparison.
    nonisolated(unsafe) static let leanTwoRowKernel = MLXFast.metalKernel(
        name: "track_gdn_lean_two_row",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out"],
        source: leanTwoRowSource, ensureRowContiguous: true)

    private static let prefetchGDNInputs =
        ProcessInfo.processInfo.environment["TRACK_GDN_INPUT_PREFETCH"] != "0"

    static let leanPrefetchSource = """
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
        float next_keys[n_per_t], next_queries[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            const int s_idx = n_per_t * dk_idx + i;
            next_keys[i] = static_cast<float>(k_[s_idx]);
            next_queries[i] = static_cast<float>(q_[s_idx]);
        }
        float next_decay = g_[hv_idx];
        float next_beta = beta_[hv_idx];
        float next_v0 = static_cast<float>(v_[dv_idx]);
        float next_v1 = static_cast<float>(v_[dv_idx + 1]);
        for (int t = 0; t < T_; ++t) {
            float keys[n_per_t], queries[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                keys[i] = next_keys[i];
                queries[i] = next_queries[i];
            }
            const float decay = next_decay;
            const float gate_beta = next_beta;
            const float value0 = next_v0;
            const float value1 = next_v1;
            if (t + 1 < T_) {
                for (int i = 0; i < n_per_t; ++i) {
                    const int s_idx = n_per_t * dk_idx + i;
                    next_keys[i] = static_cast<float>(k_[Hk * Dk + s_idx]);
                    next_queries[i] = static_cast<float>(q_[Hk * Dk + s_idx]);
                }
                next_decay = g_[Hv + hv_idx];
                next_beta = beta_[Hv + hv_idx];
                next_v0 = static_cast<float>(v_[Hv * Dv + dv_idx]);
                next_v1 = static_cast<float>(v_[Hv * Dv + dv_idx + 1]);
            }
            float kv_mem0 = 0.0f, kv_mem1 = 0.0f;
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_compensation0 = 0.0f, kv_compensation1 = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    const int s_idx = n_per_t * dk_idx + i;
                    const float key = keys[i];
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
            const float delta0 = (value0 - kv_mem0) * gate_beta;
            const float delta1 = (value1 - kv_mem1) * gate_beta;
            float out0 = 0.0f, out1 = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                const int s_idx = n_per_t * dk_idx + i;
                const float key = keys[i];
                state0[i] = state0[i] + key * delta0;
                state1[i] = state1[i] + key * delta1;
                const float query = queries[i];
                out0 += state0[i] * query;
                out1 += state1[i] * query;
            }
            out0 = simd_sum(out0);
            out1 = simd_sum(out1);
            if (dk_idx == 0) {
                y_[dv_idx] = static_cast<InT>(out0);
                y_[dv_idx + 1] = static_cast<InT>(out1);
            }
            if (CAPTURE || t == T_ - 1) {
                const uint slot = CAPTURE ? (b_idx * T_ + t) : b_idx;
                device StT* o_state = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;
                for (int i = 0; i < n_per_t; ++i) {
                    o_state[n_per_t * dk_idx + i] = static_cast<StT>(state0[i]);
                    o_state[Dk + n_per_t * dk_idx + i] = static_cast<StT>(state1[i]);
                }
            }
            q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        """

    private static let leanPrefetchKernel = MLXFast.metalKernel(
        name: "track_gdn_input_prefetch",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out"],
        source: leanPrefetchSource, ensureRowContiguous: true)

    // MARK: MLXFAST-GDNROWS -- the prefill recurrence, four rows per simdgroup
    //
    // WHY. At the prefill shape the recurrence launches grid (32, Dv/2, B*Hv)
    // = 98 304 threads and runs 1 024 sequential timesteps in 1.63 ms, which is
    // 1.59 us -- about 2 200 cycles -- per timestep for a body whose dependent
    // arithmetic is ~400 cycles. That looks like a stall, and it is not: every
    // variant of this kernel, up to eight rows per simdgroup, reports
    // maxTotalThreadsPerThreadgroup == 1024, so nothing spills and nothing
    // trades occupancy for registers. Counting the body's lane-ops instead
    // (Kahan compensation forbids FMA contraction, so it costs six operations
    // per two useful FLOPs) gives ~118 per lane per timestep, and
    // 98 304 x 1 024 x 118 x 36 layers / (40 cores x 128 ALUs x 1.4 GHz) is
    // 59.7 ms against 58.8 ms measured. The kernel is INSTRUCTION-ISSUE BOUND
    // at ~100 % of the machine's issue rate. The only thing that can make it
    // faster while staying bit-exact is issuing fewer instructions.
    //
    // Four exact reductions, measured one at a time on the real prefill shape
    // against a byte-identical control arm (noise floor 1 %):
    //
    //  1. FOUR ROWS PER SIMDGROUP instead of two (-10.9 %). The per-timestep
    //     cost that does NOT scale with the row count -- the lane's four q and
    //     four k loads, the two gate loads, the pointer arithmetic, the loop
    //     branch -- is amortised over twice as many outputs. Each row's own
    //     arithmetic sequence is untouched, so this is exact by construction,
    //     exactly as MLXFAST-GDNTILE's 1 -> 2 move was. Eight rows measured
    //     worse (-9.1 %) and sixteen much worse: past four, the shrinking
    //     thread count stops covering the memory latency.
    //  2. NO INPUT PREFETCH (part of the same -10.9 %). `49d184e` chose to hold
    //     the next timestep's q/k/v/decay/beta in registers on a reasoned,
    //     never-measured argument. Measured, it is worth nothing at two rows
    //     (-0.5 %, inside the noise floor) and it is actively harmful at four
    //     (+8.4 %): the extra live set is what finally makes the scheduler
    //     spill. Dropping it is what makes four rows available.
    //  3. PEELED KAHAN ENDS (-3.4 %). The i == 0 step runs with `kv_mem == 0`
    //     and `kv_compensation == 0`, so `corrected == product`,
    //     `next_sum == 0 + product`, and the new compensation is
    //     `product - product == 0`: four of its six operations are dead. The
    //     compensation produced by the LAST i is never read after the block, so
    //     its two subtractions are dead too. (Sign of zero: the only input that
    //     could diverge is `product == -0.0`, where the unpeeled form leaves
    //     `kv_mem == +0.0` and the peeled one `-0.0`; both give the same
    //     `delta`, and `kv_mem` is never stored, so no output can differ.)
    //  4. VECTOR LOADS (-1.1 %). The lane's four q/k entries are contiguous and
    //     8-byte aligned, the four v rows likewise, and the state row is
    //     16-byte aligned, so each becomes one load instead of four. Same
    //     values in the same registers.
    //
    // Together: 1.32 ms per launch against 1.59 ms, i.e. -17 % of the family
    // and -9.7 ms of the prefill chunk. Threadgroup depth 1/2/4/8 all measured
    // inside the noise floor of each other, so the shipped (32, 4, 1) stays.
    //
    // NOT ATTEMPTED, and it must not be: the chunkwise-parallel delta-rule form
    // that turns the scan into 64-wide matmuls is not exact -- it replaces the
    // per-timestep compensated summation with block products.

    /// The recurrence with `rps` value rows per simdgroup. Row `r`'s operations
    /// are emitted in the same order as the one-row kernel's, so every output
    /// element's arithmetic sequence is independent of `rps`.
    static func leanRowsSource(_ rps: Int) -> String {
        let rows = 0 ..< rps
        func each(_ ind: String, _ body: (Int) -> String) -> String {
            rows.map { ind + body($0) + "\n" }.joined()
        }
        var s = """
            const uint dv_idx = \(rps) * thread_position_in_grid.y;
            if (dv_idx >= Dv) { return; }
            const uint n = thread_position_in_grid.z;
            const uint b_idx = n / Hv;
            const uint hv_idx = n % Hv;
            const uint hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;
            static_assert(n_per_t == 4 && metal::is_same<StT, float>::value, "vector load shape");
            const int T_ = T;
            const uint dk_idx = thread_position_in_threadgroup.x;
            const device InT* q_ = q + b_idx * T_ * Hk * Dk + hk_idx * Dk + 4 * dk_idx;
            const device InT* k_ = k + b_idx * T_ * Hk * Dk + hk_idx * Dk + 4 * dk_idx;
            const device InT* v_ = v + b_idx * T_ * Hv * Dv + hv_idx * Dv + dv_idx;
            device InT* y_ = y + b_idx * T_ * Hv * Dv + hv_idx * Dv + dv_idx;
            const device float* g_ = g + b_idx * T_ * Hv;
            const device float* beta_ = beta + b_idx * T_ * Hv;
            const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk + 4 * dk_idx;
            // k_/q_ share the Hk*Dk pitch, v_/y_ share Hv*Dv and g_/beta_ share
            // Hv: three running offsets in place of six pointer increments.
            uint kq_off = 0, vy_off = 0, gb_off = hv_idx;

            """
        s += "float " + rows.map { "state\($0)[n_per_t]" }.joined(separator: ", ") + ";\n"
        s += each("") { r in
            "{ const float4 s4 = *reinterpret_cast<const device float4*>((const device float*)i_state + \(r) * Dk);"
                + " state\(r)[0] = s4.x; state\(r)[1] = s4.y; state\(r)[2] = s4.z; state\(r)[3] = s4.w; }"
        }
        s += """
            for (int t = 0; t < T_; ++t) {
                float keys[n_per_t], queries[n_per_t];
                {
                    const metal::vec<InT, 4> kk =
                        *reinterpret_cast<const device metal::vec<InT, 4>*>(k_ + kq_off);
                    const metal::vec<InT, 4> qq =
                        *reinterpret_cast<const device metal::vec<InT, 4>*>(q_ + kq_off);
                    for (int i = 0; i < n_per_t; ++i) {
                        keys[i] = static_cast<float>(kk[i]);
                        queries[i] = static_cast<float>(qq[i]);
                    }
                }
                const float decay = g_[gb_off];
                const float gate_beta = beta_[gb_off];
            """
        if rps == 2 || rps == 4 {
            s += "    const metal::vec<InT, \(rps)> vv ="
                + " *reinterpret_cast<const device metal::vec<InT, \(rps)>*>(v_ + vy_off);\n"
        } else {
            s += each("    ") { r in "const float vv\(r) = static_cast<float>(v_[vy_off + \(r)]);" }
        }
        s += "    float " + rows.map { "kv_mem\($0)" }.joined(separator: ", ") + ";\n"
        s += """
                {
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)

            """
        s += "        float " + rows.map { "kv_compensation\($0)" }.joined(separator: ", ") + ";\n"
        // i == 0: kv_mem and the compensation are both zero, so four of the six
        // operations are provably dead. `0.0f +` is kept so the sum's sign of
        // zero is the unpeeled form's.
        s += each("        ") { r in
            "state\(r)[0] = state\(r)[0] * decay;"
                + " kv_mem\(r) = 0.0f + state\(r)[0] * keys[0];"
                + " kv_compensation\(r) = 0.0f;"
        }
        s += "        for (int i = 1; i < n_per_t; ++i) {\n            const float key = keys[i];\n"
        // The compensation from the last i is never read again; the guard folds
        // away in the unrolled loop. A shortened loop bound would instead stop
        // the unroll, which costs far more than the two operations it saves.
        s += each("            ") { r in
            """
            {
                        state\(r)[i] = state\(r)[i] * decay;
                        auto product = state\(r)[i] * key;
                        auto corrected = product - kv_compensation\(r);
                        auto next_sum = kv_mem\(r) + corrected;
                        if (i + 1 < n_per_t) { kv_compensation\(r) = (next_sum - kv_mem\(r)) - corrected; }
                        kv_mem\(r) = next_sum;
                    }
            """
        }
        s += "        }\n    }\n"
        s += each("    ") { r in "kv_mem\(r) = simd_sum(kv_mem\(r));" }
        s += each("    ") { r in
            "const float delta\(r) = (\(rps == 2 || rps == 4 ? "static_cast<float>(vv[\(r)])" : "vv\(r)")"
                + " - kv_mem\(r)) * gate_beta;"
        }
        s += "    float " + rows.map { "out\($0) = 0.0f" }.joined(separator: ", ") + ";\n"
        s += """
                for (int i = 0; i < n_per_t; ++i) {
                    const float key = keys[i];
                    const float query = queries[i];

            """
        s += each("        ") { r in
            "state\(r)[i] = state\(r)[i] + key * delta\(r); out\(r) += state\(r)[i] * query;"
        }
        s += "    }\n"
        s += each("    ") { r in "out\(r) = simd_sum(out\(r));" }
        s += "    if (dk_idx == 0) {\n"
        s += each("        ") { r in "y_[vy_off + \(r)] = static_cast<InT>(out\(r));" }
        s += """
                }
                if (CAPTURE || t == T_ - 1) {
                    const uint slot = CAPTURE ? (b_idx * T_ + t) : b_idx;
                    device StT* o_state =
                        state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk + 4 * dk_idx;

            """
        s += each("        ") { r in
            "*reinterpret_cast<device float4*>((device float*)o_state + \(r) * Dk) ="
                + " float4(state\(r)[0], state\(r)[1], state\(r)[2], state\(r)[3]);"
        }
        s += """
                }
                kq_off += Hk * Dk; vy_off += Hv * Dv; gb_off += Hv;
            }
            """
        return s
    }

    /// Value rows per simdgroup in the prefill recurrence. Four is the measured
    /// optimum; `0` falls back to the previous prefetching two-row kernel.
    private static let prefillRows =
        Int(ProcessInfo.processInfo.environment["TRACK_GDN_PREFILL_ROWS"] ?? "") ?? 2

    nonisolated(unsafe) private static let leanRowsKernel = MLXFast.metalKernel(
        name: "track_gdn_rows",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out"],
        source: leanRowsSource(prefillRows > 0 ? prefillRows : 4), ensureRowContiguous: true)

    /// The whole deltanet core: prep + recurrence. Returns y [B,T,Hv,Dv],
    /// the conv tails and the recurrent state (per position when `capture`).
    static func gdn(
        proj: MLXArray, convState: MLXArray, convW: MLXArray,
        negExpALog: MLXArray, dtBias: MLXArray, stateIn: MLXArray,
        T: Int, capture: Bool, geometry g: GDNGeometry,
        separateBA: (b: MLXArray, a: MLXArray)? = nil
    ) -> (y: MLXArray, convOut: MLXArray, stateOut: MLXArray) {
        let B = proj.dim(0)
        let slots = capture ? B * T : B
        precondition(g.dk == 128 && g.dv == 128 && g.convDim % 128 == 0)
        let prof = TrackFastProfile.prefill != nil && T >= TrackFastProfile.minWindow
        var pt = prof ? CFAbsoluteTimeGetCurrent() : 0
        let prep: [MLXArray]
        if let separateBA {
            // The same kernel body: only the two 48-wide gate rows move from
            // offsets in the concatenated projection to their own buffers.
            prep = TrackP12Prefill.gdnPrepSplit(
                proj: proj, b: separateBA.b, a: separateBA.a, convState: convState,
                convW: convW, negExpALog: negExpALog, dtBias: dtBias, T: T,
                capture: capture, geometry: g)
        } else {
            prep = gdnPrep(
                proj: proj, convState: convState, convW: convW, negExpALog: negExpALog,
                dtBias: dtBias, T: T, capture: capture, geometry: g)
        }
        if prof { TrackFastProfile.tick("gdn.prep", &pt, prep) }
        // MLXFAST-GDNROWS: each prefill SIMD group owns `prefillRows` value rows
        // (four, measured); decode (S == 1) and the capture window keep the
        // kernels they had. Omit the unused grid rows.
        let rows: Int
        let recurrence: MLXFast.MLXFastKernel
        if prefillRows > 0 && T > 8 && !capture {
            (rows, recurrence) = (prefillRows, leanRowsKernel)
        } else if prefetchGDNInputs && T > 8 && !capture {
            (rows, recurrence) = (2, leanPrefetchKernel)
        } else if T > 1 {
            (rows, recurrence) = (2, leanTwoRowKernel)
        } else {
            (rows, recurrence) = (1, leanKernel)
        }
        let rec = recurrence(
            [prep[0], prep[1], prep[2], prep[3], prep[4], stateIn],
            template: [
                ("InT", proj.dtype), ("StT", stateIn.dtype), ("Dk", g.dk), ("Dv", g.dv),
                ("Hk", g.hk), ("Hv", g.hv), ("CAPTURE", capture), ("T", T),
            ],
            grid: (32, g.dv / rows, B * g.hv), threadGroup: (32, 4, 1),
            outputShapes: [[B, T, g.hv, g.dv], [slots, g.hv, g.dv, g.dk]],
            outputDTypes: [proj.dtype, stateIn.dtype])
        if prof { TrackFastProfile.tick("gdn.lean", &pt, rec) }
        return (rec[0], prep[5], rec[1])
    }
}
