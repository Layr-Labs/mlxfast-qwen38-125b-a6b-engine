import Foundation
import MLX

// MLXFAST-PROJSPLITK (ported from public submission 4f06f589, with the pair
// index carried through): the mixer's ORDERED split-K GEMV applied to the two
// K = 6144 dense output projections (`gdn.out_proj` and `attn.o_proj`).
//
// `TrackQuantWeight.apply` sends both through the one-row `qmv_fast` walk
// (MLX's `affine_qmv_fast`, or this tree's indexed twin of it): `N / 8`
// threadgroups of 2 simdgroups, each simdgroup owning 4 output rows and the
// WHOLE contraction, so at K = 6144 every lane folds a 12-deep dependent
// chain of quantized blocks with only `2 * N / 8` simdgroups resident.
//
// `research_split_qmv` (TrackFastMixerSplitK) does the same work with the
// chain cut into `SPLIT` pieces: simdgroup `sg` folds blocks
// `[sg * COUNT, min((sg + 1) * COUNT, NB))` of `NB = K / 512`, writes each
// block's `qdot` to threadgroup scratch, and simdgroup 0 adds them back in
// ASCENDING BLOCK ORDER before the closing `simd_sum`. The fold order, the
// `load_vector`/`qdot` calls, the scale/bias reads (through the pair index
// when the projection carries one) and the byte offsets are the ones
// `qmv_fast_impl` uses: the same arithmetic in the same order, on other threads.
//
// At RPS 2 / SPLIT 4 and K = 6144: NB = 12, COUNT = 3, four simdgroups with
// 3 blocks each and no empty range; the launch is `N / 2` threadgroups of 4.
enum TrackFastProjSplitK {
    /// Output rows per threadgroup.
    static let rows = 2
    /// K-partitions per threadgroup; one simdgroup each.
    static let split = 4
    /// Env `TRACK_PROJ_SPLITK=0` keeps the one-row walk.
    nonisolated(unsafe) static var enabled: Bool =
        ProcessInfo.processInfo.environment["TRACK_PROJ_SPLITK"] != "0"

    static let source = #"""
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const int tile = int(threadgroup_position_in_grid.y);
        threadgroup float scratch[(K / 512) * RPS * 32];
        float r[RPS];
        research_split_qmv<T, K, 16, RPS, SPLIT, true, LUT>(
            w, s, b, x, tile * RPS, sg, lane, scratch, r);
        if (sg == 0 && lane == 0) {
            for (int i = 0; i < RPS; ++i) { y[tile * RPS + i] = static_cast<T>(r[i]); }
        }
        """#

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_proj_split_k",
        inputNames: ["x", "w", "s", "b"], outputNames: ["y"], source: source,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
            + TrackFastMixerSplitK.helper,
        ensureRowContiguous: true)

    /// The split-K result for a one-token `x`, or nil when this projection is
    /// not one `qmv_fast` would have taken; the caller then keeps `TrackProj.apply`.
    static func apply(_ x: MLXArray, _ proj: TrackProj) -> MLXArray? {
        guard enabled, case .quant(let q) = proj, let biases = q.biases else { return nil }
        let k = x.dim(-1), n = q.rows
        // `x.size == k` is the one-token guard (B * S == 1). `k % 512 == 0` and
        // `n % 8 == 0` are `qmv_fast`'s own admission test; outside them the
        // plain `qmv` runs, whose fold differs.
        guard x.size == k, q.bits == 4, q.groupSize == 32, q.mode == .affine,
            k % 512 == 0, n % 8 == 0, n % rows == 0,
            q.weight.dtype == .uint32, q.scales.dtype == x.dtype, biases.dtype == x.dtype
        else { return nil }
        let useLut = q.meta != nil
        let y = kernel(
            [x.reshaped([1, k]), q.weight, useLut ? q.meta!.index : q.scales, useLut ? q.meta!.table : biases],
            template: [("T", x.dtype), ("K", k), ("RPS", rows), ("SPLIT", split), ("LUT", useLut)],
            grid: (32, (n / rows) * split, 1), threadGroup: (32, split, 1),
            outputShapes: [[1, n]], outputDTypes: [x.dtype])[0]
        return y.reshaped(Array(x.shape.dropLast()) + [n])
    }
}
