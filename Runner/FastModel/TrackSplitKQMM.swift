import Foundation
import MLX

/// MLXFAST-C2: one launch for a split-K quantized matmul, instead of MLX's two.
///
/// ## What MLX does
///
/// For `M = 1024` and a skinny `N`, `quantized_matmul` takes the split-K path
/// (`backend/metal/quantized.cpp:1124-1221`): it picks
/// `split_k = max(1, 512 / (n_tiles * m_tiles))`, has `affine_qmm_t_splitk`
/// write `split_k` partial results into a `[split_k, M, N]` **bfloat16**
/// temporary, and then launches a SECOND kernel to sum axis 0. This prefill
/// runs 216 such matmuls -- HC inject `N = 4` x96, GDN b/a `N = 48` x72,
/// shared gate `N = 1` x48 -- so it pays 432 launches where 216 would do, and
/// a serialized launch on this dependent chain costs ~46-50 us.
///
/// The host dispatch that issues both launches is outside the track's
/// `editablePaths` (only the Metal sources are), and a kernel added to the AOT
/// metallib cannot be launched from here because `MLXFast.metalKernel` only
/// sees its own JIT library. So the only way to drop the second launch is to
/// stop calling MLX for these shapes and issue one kernel that does the
/// partitions AND the fold.
///
/// ## Why it is bit-exact
///
/// Not by argument: by using the same code. `TrackSplitKMetal` is the vendored
/// JIT text copied verbatim, in the order `jit_kernels.cpp:1029-1039`
/// concatenates it for an affine quantized kernel, so each partition is
/// computed by MLX's own `qmm_t_impl` -- same `QuantizedBlockLoader`, same
/// `mlx::steel::BlockMMA`, same fp32 accumulation, same cast to bfloat16 when
/// `store_result` writes the partial.
///
/// The fold then reproduces `col_reduce_small`
/// (`kernels/reduction/reduce_col.h:4-94`), which is the kernel this reduction
/// actually reaches: `strided_reduce_general_dispatch` sends
/// `reduction_size * non_col_reductions < 32` there, and `split_k` is 8 or 16.
/// Its accumulator type comes from `remap_reduce_types`, which for `sum` over
/// bfloat16 returns `{bfloat16, bfloat16}`, so the fold is in bfloat16, not
/// fp32, and `Sum` is `op(a, b) = a + b` applied as `op(partial, total)`.
///
/// The fold's shape is not a plain left-to-right sum. With `n_reads = 4` and
/// `reduction_stride = M * N`, `col_reduce_small` launches `(32, Y, 1)` threads
/// where `Y = min(8, split_k)`; each `y` walks `r = y, y + Y, ...` from
/// `Op::init`, then lane 0 folds the `Y` lane totals:
///
///     t[y] = 0;  for (r = y; r < split_k; r += Y)  t[y] = p[r] + t[y]
///     acc = t[0];  for (j = 1; j < Y; j++)  acc = t[j] + acc
///
/// For `split_k = 8` that degenerates to `((p0 + 0) + p1) + ... + p7`; for
/// `split_k = 16` each lane total is `p[y+8] + (p[y] + 0)` first. The leading
/// `+ 0` is the identity for every finite bfloat16 value, but it does turn a
/// `-0` partial into `+0`.
///
/// **`split_k = 8` cannot discriminate this structure**, because at 8 each lane
/// holds one term and the lane fold degenerates to the same left fold a naive
/// implementation would produce. Only the `split_k = 16` families can tell a
/// correct fold from a wrong one, which is why the exactness test covers all
/// three families rather than the cheapest one.
///
/// ## Why the tile is 16x16 and not MLX's 32x32
///
/// `split_k` exists to MANUFACTURE threadgroups: `512 / (n_tiles * m_tiles)` is
/// chosen to bring the grid back up to ~512. Folding it into this kernel's loop
/// therefore trades away exactly the parallelism it buys -- at MLX's own 32x32
/// tile the grid is 64 threadgroups for `N = 48` and 32 for `N = 4`, and the
/// fused kernel measures **+2.48 ms** of prefill (t = 59.8), i.e. slower.
///
/// Retiling is arithmetic-neutral -- `BlockMMA` has `TM = BM / (8 * WM)` and
/// `TN = BN / (8 * WN)` with `WM = WN = 2`, so BM and BN only have to be
/// multiples of 16, and changing them changes which simdgroup owns an output
/// element, not the order in which any element accumulates over K. 16x16 gives
/// `N = 48` 192 threadgroups and measures **-2.08 ms** (t = -3.87, n = 3),
/// and 16x32 sits between the two at -0.53 ms. All of 16x16, 16x32, 32x16 and
/// 32x32 pass the exactness test.
///
/// Running G partitions concurrently inside one threadgroup was tried and is
/// worse, monotonically: G = 4 costs +15.0 ms and G = 8 costs +21.9 ms, while
/// all of G = 1, 2, 4, 8 stay bit-exact. The barriers inside `qmm_t_impl` then
/// synchronise all `128 * G` threads rather than one group's 128, coupling the
/// groups into a single lock-step pipeline; and adding threads cannot recover
/// what was lost anyway, because the partition loop is DEPENDENT where MLX's
/// 512 threadgroups are INDEPENDENT.
///
/// So this is a small win with a structural ceiling: on split-K, removing
/// launches is self-limiting, because MLX's launches are its parallelism.
enum TrackSplitKQMM {
    /// `TRACK_SPLITK_QMM=0` routes every shape back to `quantizedMatmul`.
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_SPLITK_QMM"] != "0"

    /// Output tile. 16x16 measured best; see the type comment.
    private static let BM = 16
    private static let BN = 16

    /// MLX's own choice of `split_k` for a transposed affine quantized matmul,
    /// reproduced from `quantized.cpp:1138-1159`. Returns nil when MLX would
    /// not take the split-K path at all, which is also our bail-out. Note this
    /// uses MLX's 32x32 tiling, not ours: `split_k` has to match what the
    /// reduce would have folded, whatever tile we then compute it with.
    static func splitK(M: Int, N: Int, K: Int, groupSize: Int) -> Int? {
        let bm = 32, bn = 32
        let nTiles = (N + bn - 1) / bn
        let mTiles = (M + bm - 1) / bm
        var splitK = max(1, 512 / (nTiles * mTiles))
        let kAlign = groupSize > 32 ? groupSize : 32
        splitK = min(splitK, K / kAlign)
        while splitK > 1 && K % (splitK * kAlign) != 0 {
            splitK -= 1
        }
        return splitK > 1 ? splitK : nil
    }

    private struct Key: Hashable {
        let M: Int, N: Int, K: Int, groupSize: Int, bits: Int, splitK: Int
    }

    nonisolated(unsafe) private static var kernels: [Key: MLXFast.MLXFastKernel] = [:]

    private static func kernel(_ key: Key) -> MLXFast.MLXFastKernel {
        if let k = kernels[key] { return k }
        let kp = key.K / key.splitK
        // `qmm_t_impl` takes `const constant int&` for K/N/M/K_eff, which a
        // literal cannot bind to, and Metal allows `constant` variables only at
        // program scope. So the dimensions ride in a program-scope array
        // appended AFTER the verbatim vendored text, which stays untouched.
        let dims = """

            constant int tsk_dims[4] = { \(key.K), \(key.N), \(key.M), \(kp) };
            """
        let k = MLXFast.metalKernel(
            name: "track_qmm_t_splitk_fused_m\(key.M)_n\(key.N)_k\(key.K)_spk\(key.splitK)",
            inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y", "partials"],
            source: source(
                splitK: key.splitK, groupSize: key.groupSize, bits: key.bits,
                alignedN: key.N % BN == 0),
            header: TrackSplitKMetal.header + dims,
            ensureRowContiguous: true)
        kernels[key] = k
        return k
    }

    /// `[M, K] x [N, K]^T -> [M, N]`, or nil when this is not a shape we own.
    static func matmul(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray,
        groupSize: Int, bits: Int
    ) -> MLXArray? {
        guard enabled, StreamOrDevice.default.stream == Stream.gpu,
            x.ndim == 2, w.ndim == 2, x.dtype == .bfloat16,
            scales.dtype == .bfloat16, biases.dtype == .bfloat16,
            w.dtype == .uint32, groupSize == 32, bits == 4
        else { return nil }
        let M = x.dim(0), K = x.dim(1), N = w.dim(0)
        // MLX only reaches `qmm_splitk` when `M >= get_qmv_batch_limit(K, N)`
        // and the weights are transposed and unbatched
        // (`quantized.cpp:2004-2014`); below that limit it runs a `qmv`, whose
        // arithmetic is NOT the split-K fold. The limit is a device table
        // (`quantized.cpp:89-140`) whose largest entry is 33, so requiring 64
        // is safe without reproducing the table: prefill is M = 1024, and the
        // decode and MTP-verify windows (M <= 8) keep going to MLX, which is
        // what they already do.
        guard M >= 64, M % BM == 0,
            let splitK = splitK(M: M, N: N, K: K, groupSize: groupSize),
            scales.dim(0) == N, scales.dim(1) == K / groupSize
        else { return nil }
        let key = Key(M: M, N: N, K: K, groupSize: groupSize, bits: bits, splitK: splitK)
        let nTiles = (N + BN - 1) / BN
        let mTiles = (M + BM - 1) / BM
        return kernel(key)(
            [w, scales, biases, x],
            // MLX dispatches THREADGROUPS (n_tiles, m_tiles, split_k) with
            // (32, 2, 2) threads; a custom kernel dispatches THREADS, and the
            // split_k dimension collapses into this kernel's own loop.
            grid: (nTiles * 32, mTiles * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[M, N], [splitK, M, N]],
            outputDTypes: [.bfloat16, .bfloat16])[0]
    }

    static func source(splitK: Int, groupSize: Int, bits: Int, alignedN: Bool) -> String {
        let foldY = min(8, splitK)
        return #"""
            constexpr int BM = \#(BM);
            constexpr int BK = 32;
            constexpr int BN = \#(BN);
            constexpr int SPLIT_K = \#(splitK);
            constexpr int FOLD_Y = \#(foldY);
            constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);
            constexpr int pack_factor = get_pack_factor<\#(bits), 8>();
            constexpr int bytes_per_pack = get_bytes_per_pack<\#(bits)>();

            static_assert(BM % 16 == 0 && BN % 16 == 0, "BlockMMA needs TM, TN >= 1");

            const int N = tsk_dims[1];
            const int M = tsk_dims[2];
            const int kp = tsk_dims[3];
            const int part_stride = M * N;

            threadgroup bfloat16_t Xs[BM * BK_padded];
            threadgroup bfloat16_t Ws[BN * BK_padded];

            const uint3 tid = threadgroup_position_in_grid;
            const uint lid = thread_index_in_threadgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;

            // One partition per iteration, with the pointers offset exactly as
            // `affine_qmm_t_splitk` offsets its own from `tid.z`.
            for (int z = 0; z < SPLIT_K; z++) {
              const int k_start = z * kp;
              auto wl = (const device uint8_t*)w + k_start * bytes_per_pack / pack_factor;
              threadgroup_barrier(mem_flags::mem_threadgroup);
              qmm_t_impl<bfloat16_t, \#(groupSize), \#(bits), \#(alignedN ? "true" : "false"), BM, BK, BN>(
                  (const device uint32_t*)wl,
                  scales + k_start / \#(groupSize),
                  biases + k_start / \#(groupSize),
                  x + k_start,
                  partials + z * part_stride,
                  Xs, Ws,
                  tsk_dims[0], tsk_dims[1], tsk_dims[2], tsk_dims[3],
                  tid, lid, simd_gid, simd_lid);
            }

            // This threadgroup owns its (m_tile, n_tile) region across EVERY
            // partition, so the fold needs no cross-threadgroup visibility --
            // only this threadgroup's own writes, which the device barrier
            // orders. That is what makes one launch possible at all.
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            const int y_row = int(tid.y) * BM;
            const int y_col = int(tid.x) * BN;
            const int rows = min(BM, M - y_row);
            const int cols = min(BN, N - y_col);
            const uint tg_threads = 32 * 2 * 2;
            for (uint idx = lid; idx < uint(rows * cols); idx += tg_threads) {
              const int i = int(idx) / cols;
              const int j = int(idx) - i * cols;
              const int off = (y_row + i) * N + (y_col + j);
              // `col_reduce_small`: FOLD_Y lane totals from Op::init, each
              // walking r = y, y + FOLD_Y, ..., then lane 0 folds the lanes.
              bfloat16_t t[FOLD_Y];
              for (int yy = 0; yy < FOLD_Y; yy++) {
                bfloat16_t acc = bfloat16_t(0);
                for (int r = yy; r < SPLIT_K; r += FOLD_Y) {
                  acc = partials[r * part_stride + off] + acc;
                }
                t[yy] = acc;
              }
              bfloat16_t total = t[0];
              for (int jj = 1; jj < FOLD_Y; jj++) {
                total = t[jj] + total;
              }
              y[off] = total;
            }
            """#
    }
}
