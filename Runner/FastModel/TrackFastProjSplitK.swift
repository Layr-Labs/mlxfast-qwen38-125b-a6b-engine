import Foundation
import MLX

// MLXFAST-PROJSPLITK: the mixer's ORDERED split-K GEMV, applied to the two
// K = 6144 dense output projections (`gdn.out_proj` and `attn.o_proj`).
//
// `TrackQuantWeight.apply` sends both of them through `quantizedMM` ->
// MLX `affine_qmv_fast`, which lays out `N / 8` threadgroups of 2 simdgroups
// and gives each simdgroup 4 output rows and the WHOLE contraction: every lane
// folds `K / 512` quantized blocks into one register, serially. At K = 6144
// that is a 12-deep dependent chain per lane with only `2 * N / 8` simdgroups
// resident to hide it.
//
// `research_split_qmv` (TrackFastMixerSplitK) already does exactly this shape
// of work with the chain cut into `SPLIT` pieces: simdgroup `sg` folds only
// blocks `[sg * COUNT, min((sg + 1) * COUNT, NB))` of `NB = K / 512`, writes
// each block's `qdot` to threadgroup scratch, and simdgroup 0 then adds them
// back in ASCENDING BLOCK ORDER before the closing `simd_sum`. The fold order,
// the `load_vector`/`qdot` calls, the scale/bias indices and the byte offsets
// are the ones `qmv_fast_impl` uses, so the arithmetic is the same arithmetic
// in the same order -- only the thread that performs it moves.
//
// At RPS 2 / SPLIT 4 and K = 6144: NB = 12, COUNT = 3, so the four simdgroups
// own 3 blocks each with no empty range, and the launch is `N / 2`
// threadgroups of 4 simdgroups.
enum TrackFastProjSplitK {
    /// Output rows per threadgroup.
    static let rows = 2
    /// K-partitions per threadgroup; one simdgroup each.
    static let split = 4

    static let source = #"""
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int tile = int(threadgroup_position_in_grid.y);
        threadgroup float scratch[(K / 512) * RPS * 32];
        float r[RPS];
        research_split_qmv<T, K, 16, RPS, SPLIT, true>(
            w, s, b, x, tile * RPS, sg, lane, scratch, r);
        if (sg == 0 && lane == 0) {
            for (int i = 0; i < RPS; ++i) { y[tile * RPS + i] = static_cast<T>(r[i]); }
        }
        """#

    static let kernel = MLXFast.metalKernel(name: "track_proj_split_k",
        inputNames: ["x", "w", "s", "b"], outputNames: ["y"], source: source,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
            + TrackFastMixerSplitK.helper,
        ensureRowContiguous: true)

    /// The split-K result for a one-token `x`, or nil when this projection is
    /// not one `affine_qmv_fast` would have taken -- the caller then keeps the
    /// shipped `TrackProj.apply` path, whose rounding this only reproduces
    /// for the geometry the guards admit.
    static func apply(_ x: MLXArray, _ proj: TrackProj) -> MLXArray? {
        guard case .quant(let q) = proj, let biases = q.biases else { return nil }
        let k = x.dim(-1), n = q.rows
        // `x.size == k` is the one-token guard (B * S == 1). `k % 512 == 0`
        // and `n % 8 == 0` are `affine_qmv_fast`'s own admission test
        // (`qmv_fast_k_alignment(4) == 512`, `bn == 8`): outside them
        // `quantizedMM` dispatches the plain `qmv`, whose fold differs.
        guard x.size == k, q.bits == 4, q.groupSize == 32, q.mode == .affine,
            k % 512 == 0, n % 8 == 0, n % rows == 0,
            q.weight.dtype == .uint32, q.scales.dtype == x.dtype, biases.dtype == x.dtype
        else { return nil }
        let y = kernel(
            [x.reshaped([1, k]), q.weight, q.scales, biases],
            template: [("T", x.dtype), ("K", k), ("RPS", rows), ("SPLIT", split)],
            grid: (32, (n / rows) * split, 1), threadGroup: (32, split, 1),
            outputShapes: [[1, n]], outputDTypes: [x.dtype])[0]
        return y.reshaped(Array(x.shape.dropLast()) + [n])
    }
}

// MLXFAST-TAG-pskr7 (20260923-092217-7): build tag. The code change in this draw is the
// new kernel above and the two call sites it is wired into.
