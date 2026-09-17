import Foundation
import MLX

// Partition HC input blocks across SIMD groups while retaining each lane's
// ascending block fold and the final simd_sum. No target weight changes.
enum TrackFastMixerSplitK {
    // Four partitions in production; zero selects the original A/B control.
    // MLXFAST-SPLIT5: K-partitions per threadgroup in the split-K mixer.
    // `research_split_qmv` cuts the contraction into `NB = K / (V * 32)` whole
    // quantized blocks and hands simdgroup `sg` the range
    // `[sg * COUNT, min((sg + 1) * COUNT, NB))` with
    // `COUNT = (NB + SPLIT - 1) / SPLIT`, so SPLIT sets how evenly those blocks
    // divide. On the down half V is 16, BLOCK 512 and NB = K / 512 = 5; at
    // SPLIT 4 that is COUNT 2 and the ranges are 2, 2, 1 and EMPTY -- the
    // fourth simdgroup owns no block and the threadgroup's critical path is
    // two blocks. At 5 it is COUNT 1 and each of the five simdgroups owns
    // exactly one block. On the inject half V is 8, NB = K / 256 = 10, and
    // COUNT goes 3 (3, 3, 3, 1) -> 2 (2, 2, 2, 2, 2).
    // The reduction is unchanged: under ORDERED it sums `scratch[(b * R + r) *
    // 32 + lane]` over `b` in `[0, NB)` in block order, and the ranges cover
    // `[0, NB)` exactly once at either value, so the same partial products are
    // added in the same order. Threadgroup scratch is
    // `(K / 512) * RPS * 32` and `(K / 256) * 32` floats under ORDERED, i.e.
    // independent of SPLIT. The threadgroup grows from 32 x 4 to 32 x 5
    // threads; the threadgroup COUNT is `n / rows + hc` either way, because the
    // grid y extent carries the same factor.
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
}
