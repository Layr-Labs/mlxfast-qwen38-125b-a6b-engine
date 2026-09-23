import Foundation
import MLX

// Partition HC input blocks across SIMD groups while retaining each lane's
// ascending block fold and the final simd_sum. No target weight changes.
enum TrackFastMixerSplitK {
    // Four partitions in production; zero selects the original A/B control.
    nonisolated(unsafe) static var split = 5
    static let helper = #"""
        template <typename T, int K, int V, int R, int SPLIT, bool ORDERED>
        METAL_FUNC void research_split_qmv(
            const device uint32_t* w, const device T* scales,
            const device T* biases, const device T* x, int row,
            uint sg, uint lane, threadgroup float* scratch,
            thread float (&result)[R]) {
            constexpr int BLOCK = V * 32;
            constexpr int NB = K / BLOCK;
            static_assert(K % BLOCK == 0, "full quantized blocks required");
            constexpr int COUNT = (NB + SPLIT - 1) / SPLIT;
            float partial[R];
            for (int r = 0; r < R; ++r) { partial[r] = 0; result[r] = 0; }
            for (int b = int(sg) * COUNT; b < min((int(sg) + 1) * COUNT, NB); ++b) {
                float xv[V];
                const int column = b * BLOCK + int(lane) * V;
                float sum = load_vector<T, float, V, 4>(x + column, xv);
                for (int r = 0; r < R; ++r) {
                    const device uint8_t* wp = (const device uint8_t*)w
                        + (row + r) * (K / 2) + column / 2;
                    const int qi = (row + r) * (K / 32) + column / 32;
                    float v = qdot<float, V, 4>(wp, xv, float(scales[qi]), float(biases[qi]), sum);
                    if constexpr (ORDERED) { scratch[(b * R + r) * 32 + lane] = v; }
                    else { partial[r] += v; }
                }
            }
            if constexpr (!ORDERED) {
                for (int r = 0; r < R; ++r) { scratch[(sg * R + r) * 32 + lane] = partial[r]; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                for (int b = 0; b < (ORDERED ? NB : SPLIT); ++b) {
                    for (int r = 0; r < R; ++r) { result[r] += scratch[(b * R + r) * 32 + lane]; }
                }
                for (int r = 0; r < R; ++r) { result[r] = simd_sum(result[r]); }
            }
        }
        """#
    static let fusedSource = #"""
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int tile = int(threadgroup_position_in_grid.y);
        constexpr int DN = ND / RPS;
        constexpr int DOWN_SCRATCH = (ORDERED ? K / 512 : SPLIT) * RPS * 32;
        constexpr int INJ_SCRATCH = (ORDERED ? K / 256 : SPLIT) * 32;
        threadgroup float scratch[DOWN_SCRATCH > INJ_SCRATCH ? DOWN_SCRATCH : INJ_SCRATCH];
        if (tile < DN) {
            float r[RPS];
            research_split_qmv<T, K, 16, RPS, SPLIT, ORDERED>(
                wd, sd, bd, x, tile * RPS, sg, lane, scratch, r);
            if (sg == 0 && lane == 0) {
                for (int i = 0; i < RPS; ++i) {
                    T l = static_cast<T>(r[i]);
                    lo[tile * RPS + i] = l;
                    act[tile * RPS + i] = mlx_silu(l);
                }
            }
        } else if (HAS_INJECT) {
            float r[1];
            research_split_qmv<T, K, 8, 1, SPLIT, ORDERED>(
                wi, si, bi, x, tile - DN, sg, lane, scratch, r);
            if (sg == 0 && lane == 0) { inj[tile - DN] = static_cast<T>(r[0]); }
        }
        """#
    static let fusedKernel = MLXFast.metalKernel(name: "track_split_k_mixer",
        inputNames: ["x", "wd", "sd", "bd", "wi", "si", "bi"],
        outputNames: ["lo", "act", "inj"], source: fusedSource,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + helper,
        ensureRowContiguous: true)

    static func apply(_ x: MLXArray, down: TrackQuantWeight, inject: TrackQuantWeight?) -> [MLXArray] {
        let k = x.size, n = down.rows, hc = inject?.rows ?? 4
        let rows = 2, partitions = split
        let inj = inject ?? down
        precondition(x.shape == [1, k] && k % 512 == 0 && n % rows == 0 && partitions > 0)
        return fusedKernel([x, down.weight, down.scales, down.biases!, inj.weight, inj.scales, inj.biases!],
            template: [("T", x.dtype), ("K", k), ("ND", n), ("RPS", rows), ("SPLIT", partitions),
                       ("ORDERED", true), ("HAS_INJECT", inject != nil)],
            grid: (32, (n / rows + (inject != nil ? hc : 0)) * partitions, 1), threadGroup: (32, partitions, 1),
            outputShapes: [[1, n], [1, n], [1, hc]], outputDTypes: [x.dtype, x.dtype, x.dtype])
    }

    // One-token inject+RMS and the split-K down/inject GEMV in one launch.
    // The RMS tree is the 640-thread `injectNorm` layout (20 virtual simdgroups,
    // 4 elements each, zero-padded `simd_sum` over `local_sums`), played by the
    // SPLIT simdgroups this threadgroup already has. Every tile recomputes the
    // four inv_means; there is no cross-tile wait. Tile 0 also writes `stream`
    // and `normed` for the up-mix that still follows.
    static let injectHelpers = #"""
        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector(const thread T* x, thread U* x_thread) {
            static_assert(bits == 4, "thread activation load is the 4-bit qmv path");
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {
                sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
                x_thread[i] = x[i];
                x_thread[i + 1] = x[i + 1] / 16.0f;
                x_thread[i + 2] = x[i + 2] / 256.0f;
                x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
            return sum;
        }

        template <typename T, int K, int V, int R, int SPLIT, bool ORDERED, bool TILE, bool STREAM_INJECT, int H, int HC>
        METAL_FUNC void research_split_qmv_stream(
            const device uint32_t* w, const device T* scales, const device T* biases,
            const device T* residual, const device T* outv, const device T* injv, const device T* scale,
            thread float* invs, int row, uint sg, uint lane, threadgroup float* scratch,
            thread float (&result)[R]) {
            constexpr int BLOCK = V * 32;
            constexpr int NB = K / BLOCK;
            static_assert(K % BLOCK == 0, "full quantized blocks required");
            constexpr int COUNT = (NB + SPLIT - 1) / SPLIT;
            float partial[R];
            for (int r = 0; r < R; ++r) { partial[r] = 0; result[r] = 0; }
            for (int b = int(sg) * COUNT; b < min((int(sg) + 1) * COUNT, NB); ++b) {
                const int column = b * BLOCK + int(lane) * V;
                const int hc = column / H;
                const int d0 = column - hc * H;
                const T inj_t = STREAM_INJECT ? injv[hc] : T(0);
                const float inv = invs[hc];
                thread T tmp[V];
                for (int i = 0; i < V; ++i) {
                    const int d = d0 + i;
                    T r0 = TILE ? residual[d] : residual[hc * H + d];
                    if (STREAM_INJECT) {
                        T sp = outv[d] * inj_t;
                        r0 = r0 + sp;
                    }
                    T n = static_cast<T>(static_cast<float>(r0) * inv);
                    tmp[i] = n * scale[hc * H + d];
                }
                float xv[V];
                float sum = load_vector<T, float, V, 4>(tmp, xv);
                for (int r = 0; r < R; ++r) {
                    const device uint8_t* wp = (const device uint8_t*)w
                        + (row + r) * (K / 2) + column / 2;
                    const int qi = (row + r) * (K / 32) + column / 32;
                    float v = qdot<float, V, 4>(wp, xv, float(scales[qi]), float(biases[qi]), sum);
                    if constexpr (ORDERED) { scratch[(b * R + r) * 32 + lane] = v; }
                    else { partial[r] += v; }
                }
            }
            if constexpr (!ORDERED) {
                for (int r = 0; r < R; ++r) { scratch[(sg * R + r) * 32 + lane] = partial[r]; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                for (int b = 0; b < (ORDERED ? NB : SPLIT); ++b) {
                    for (int r = 0; r < R; ++r) { result[r] += scratch[(b * R + r) * 32 + lane]; }
                }
                for (int r = 0; r < R; ++r) { result[r] = simd_sum(result[r]); }
            }
        }
        """#

    static let injectSource = #"""
        constexpr int N_READS = 4;
        constexpr int RMS_GROUPS = H / (32 * N_READS);
        static_assert(H % (32 * N_READS) == 0, "rms groups cover H");
        static_assert(RMS_GROUPS % SPLIT == 0, "split simdgroups cover the rms tree");
        static_assert(K == HC * H, "mixer K is the hyper stream");
        constexpr int SLICES = RMS_GROUPS / SPLIT;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int tile = int(threadgroup_position_in_grid.y);
        threadgroup float local_sums[HC * 32];
        thread float invs[HC];
        // All four streams' partials land before one barrier, so their loads
        // overlap; each stream keeps its own 32-slot fold (zero-padded) and the
        // same per-lane order as the 640-thread reference.
        if (sg == 0 && lane >= (uint)RMS_GROUPS) {
            for (int hc = 0; hc < HC; ++hc) { local_sums[hc * 32 + lane] = 0; }
        }
        for (int hc = 0; hc < HC; ++hc) {
            const T inj_t = STREAM_INJECT ? inj_vec[hc] : T(0);
            for (int sl = 0; sl < SLICES; ++sl) {
                const int g = int(sg) + sl * SPLIT;
                const uint lid = (uint)g * 32u + lane;
                float acc = 0.0f;
                for (int i = 0; i < N_READS; ++i) {
                    const uint d = lid * N_READS + (uint)i;
                    T r0 = TILE ? residual[d] : residual[(uint)hc * H + d];
                    if (STREAM_INJECT) {
                        T sp = outv[d] * inj_t;
                        r0 = r0 + sp;
                    }
                    const float xf = static_cast<float>(r0);
                    acc += xf * xf;
                }
                acc = simd_sum(acc);
                if (lane == 0) { local_sums[hc * 32 + g] = acc; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int hc = 0; hc < HC; ++hc) {
            const float total = simd_sum(local_sums[hc * 32 + lane]);
            invs[hc] = metal::precise::rsqrt(total / (float)H + as_type<float>((uint)EPS_BITS));
        }
        if (tile == 0) {
            for (int hc = 0; hc < HC; ++hc) {
                const T inj_t = STREAM_INJECT ? inj_vec[hc] : T(0);
                const float inv = invs[hc];
                for (int sl = 0; sl < SLICES; ++sl) {
                    const int g = int(sg) + sl * SPLIT;
                    const uint lid = (uint)g * 32u + lane;
                    for (int i = 0; i < N_READS; ++i) {
                        const uint d = lid * N_READS + (uint)i;
                        const uint dst = (uint)hc * H + d;
                        T r0 = TILE ? residual[d] : residual[dst];
                        if (STREAM_INJECT) {
                            T sp = outv[d] * inj_t;
                            r0 = r0 + sp;
                        }
                        stream[dst] = r0;
                        T n = static_cast<T>(static_cast<float>(r0) * inv);
                        normed[dst] = n * scale[dst];
                    }
                }
            }
        }
        constexpr int DN = ND / RPS;
        constexpr int DOWN_SCRATCH = (ORDERED ? K / 512 : SPLIT) * RPS * 32;
        constexpr int INJ_SCRATCH = (ORDERED ? K / 256 : SPLIT) * 32;
        threadgroup float scratch[DOWN_SCRATCH > INJ_SCRATCH ? DOWN_SCRATCH : INJ_SCRATCH];
        if (tile < DN) {
            float r[RPS];
            research_split_qmv_stream<T, K, 16, RPS, SPLIT, ORDERED, TILE, STREAM_INJECT, H, HC>(
                wd, sd, bd, residual, outv, inj_vec, scale, invs, tile * RPS, sg, lane, scratch, r);
            if (sg == 0 && lane == 0) {
                for (int i = 0; i < RPS; ++i) {
                    T l = static_cast<T>(r[i]);
                    lo[tile * RPS + i] = l;
                    act[tile * RPS + i] = mlx_silu(l);
                }
            }
        } else if (HAS_INJECT) {
            float r[1];
            research_split_qmv_stream<T, K, 8, 1, SPLIT, ORDERED, TILE, STREAM_INJECT, H, HC>(
                wi, si, bi, residual, outv, inj_vec, scale, invs, tile - DN, sg, lane, scratch, r);
            if (sg == 0 && lane == 0) { inj[tile - DN] = static_cast<T>(r[0]); }
        }
        """#

    static let injectKernel = MLXFast.metalKernel(
        name: "track_split_k_mixer_inject",
        inputNames: ["residual", "outv", "inj_vec", "scale", "wd", "sd", "bd", "wi", "si", "bi"],
        outputNames: ["stream", "normed", "lo", "act", "inj"],
        source: injectSource,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + helper + injectHelpers,
        ensureRowContiguous: true)

    /// Nil when the one-token fused launch does not apply. `streamInject` is the
    /// residual add (`out * inject`), independent of the mixer's own inject GEMV.
    static func injectDown(
        residual: MLXArray, out: MLXArray?, injectVec: MLXArray?, scale: MLXArray,
        down: TrackQuantWeight, inject: TrackQuantWeight?,
        hcCount: Int, hidden: Int, eps: Float, tile: Bool
    ) -> (stream: MLXArray, normed: MLXArray, lo: MLXArray, act: MLXArray, inj: MLXArray)? {
        let groups = hidden / 128
        let w = hcCount * hidden
        guard residual.dim(0) == 1, residual.dim(1) == 1, residual.dtype == .bfloat16,
            scale.dtype == .bfloat16, scale.size == w,
            down.bits == 4, down.groupSize == 32, down.biases != nil,
            hidden % 128 == 0, groups > 0, split > 0, groups % split == 0,
            hcCount > 0, w % 512 == 0, down.rows % 2 == 0,
            residual.dim(2) == (tile ? hidden : w),
            StreamOrDevice.default.stream === Stream.gpu
        else { return nil }
        let streamInject = out != nil
        if streamInject {
            guard let out, let injectVec, out.dtype == .bfloat16, injectVec.dtype == .bfloat16,
                out.dim(0) == 1, out.dim(1) == 1, out.dim(2) == hidden, injectVec.dim(-1) == hcCount
            else { return nil }
        }
        if inject != nil {
            guard let inject, inject.bits == 4, inject.groupSize == 32, inject.biases != nil,
                inject.rows == hcCount else { return nil }
        }
        let rows = 2
        let partitions = split
        let injW = inject ?? down
        let outs = injectKernel(
            [residual, out ?? residual, injectVec ?? residual, scale,
             down.weight, down.scales, down.biases!, injW.weight, injW.scales, injW.biases!],
            template: [
                ("T", residual.dtype), ("K", w), ("ND", down.rows), ("RPS", rows),
                ("SPLIT", partitions), ("ORDERED", true), ("HAS_INJECT", inject != nil),
                ("TILE", tile), ("STREAM_INJECT", streamInject), ("H", hidden), ("HC", hcCount),
                ("EPS_BITS", Int(eps.bitPattern)),
            ],
            grid: (32, (down.rows / rows + (inject != nil ? hcCount : 0)) * partitions, 1),
            threadGroup: (32, partitions, 1),
            outputShapes: [[1, 1, w], [1, 1, w], [1, down.rows], [1, down.rows], [1, hcCount]],
            outputDTypes: [residual.dtype, residual.dtype, residual.dtype, residual.dtype, residual.dtype])
        return (outs[0], outs[1], outs[2], outs[3], outs[4])
    }
}

// MLXFAST-TAG-spl5r1 (20260923-024828-1): build tag. The code change in this draw is the
// single constant above.
