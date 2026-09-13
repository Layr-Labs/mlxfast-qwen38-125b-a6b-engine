// MLXFAST-V2TWOROW: serial-semantic two-row execution, ON by default.
//
// Serial-semantic two-row execution for the three launches that carry the
// largest SHAPE penalty in a depth-1 verify window: the MoE router, the
// hyper-connection mixer down+inject, and the dense quantized GEMV.
//
// The contract is stronger than "matches the S = 2 reference": each logical row
// here performs EXACTLY the arithmetic the SERIAL S = 1 kernel performs for that
// row -- same lanes, same block walk, same `qdot`, same accumulation order, same
// closing `simd_sum`. What the two rows share is the WEIGHT STREAM: the device
// pointer is walked once and both rows' `qdot`s read through it, so the DRAM
// traffic of a two-row launch equals a one-row launch's while every row's
// reduction tree is untouched. Nothing is held in thread registers that the
// reference reads through a device pointer (the `mlx-dec3` round proved that a
// register-held replica of `qdot` is NOT bit-identical at any FP pragma).
//
// Each kernel carries a marker-file KILL SWITCH (on by default) so one build
// serves both arms of a local A/B; a measured box has no such file.
//
// The dense two-row GEMV below is deliberately NOT wired into the model: under
// the strong per-row contract it MEASURES WORSE than the wide S >= 2 kernel it
// would replace (that launch is already at ~480 GB/s and the wide kernel's eight
// lanes per row buy more than the duplicate DRAM pass they cost), so the dense
// decode GEMVs keep the wide path. It is kept here because the bit-comparison
// harness uses it to prove `qmv_fast_reg2` against MLX's own decode kernel.

import Foundation
import MLX
import MLXFast

enum TrackVerify2TwoRow {

    // MARK: shared helper -- `qmv_fast_reg` with two activation vectors.

    static let twoRowHelpers = #"""

        // qmv_fast_reg with TWO activation vectors over ONE weight walk.
        // Row r's accumulator sees exactly the sequence of qdot results the
        // one-vector helper produces for that row: same block order, same
        // scale/bias reads, same device-pointer operands, same simd_sum.
        template <typename T, int group_size, int bits, int results_per_simdgroup = 4>
        METAL_FUNC void qmv_fast_reg2(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            const device T* x1,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&r0)[results_per_simdgroup],
            thread float (&r1)[results_per_simdgroup]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x0_thread[values_per_thread];
          thread U x1_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) { r0[row] = 0; r1[row] = 0; }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x0 += simd_lid * values_per_thread;
          x1 += simd_lid * values_per_thread;
          for (int k = 0; k < in_vec_size; k += block_size) {
            U sum0 = load_vector<T, U, values_per_thread, bits>(x0, x0_thread);
            U sum1 = load_vector<T, U, values_per_thread, bits>(x1, x1_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              r0[row] += qdot<U, values_per_thread, bits>(wl, x0_thread, s, b, sum0);
              r1[row] += qdot<U, values_per_thread, bits>(wl, x1_thread, s, b, sum1);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x0 += block_size;
            x1 += block_size;
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            r0[row] = simd_sum(r0[row]);
            r1[row] = simd_sum(r1[row]);
          }
        }

        // `track_inject_qmv_row<ROW>` with two activation vectors, same rule.
        template <typename T, int group_size, int bits, int KD, int ROW>
        METAL_FUNC void track_inject_qmv_row2(
            const device uint32_t* wi,
            const device T* si,
            const device T* bi,
            const device T* n0,
            const device T* n1,
            device T* inj,
            const int HCW,
            uint simd_lid) {
          float r0[1];
          float r1[1];
          qmv_fast_reg2<T, group_size, bits, 1>(wi, si, bi, n0, n1, KD, ROW, simd_lid, r0, r1);
          if (simd_lid == 0) {
            inj[ROW] = static_cast<T>(r0[0]);
            inj[HCW + ROW] = static_cast<T>(r1[0]);
          }
        }
        """#

    // MARK: 1. router -- two rows, one bf16 weight walk.
    //
    // The reference (`track_router_gemv`, RPS = 1 for [512 x 2560]) is MLX's
    // float `gemv` walk. Both rows keep `result[tm] += inter[tn] * v_coeff[tn]`
    // in the same order over the same `inter` values; only `v_coeff` differs.

    static let router2RowSource = """
        constexpr int TM = RPS, TN = 4, SN = 32, blockM = 4 * RPS, blockN = 128;
        const int tid_x = (int)threadgroup_position_in_grid.x;
        const int simd_gid = (int)simdgroup_index_in_threadgroup;
        const int simd_lid = (int)thread_index_in_simdgroup;
        float r0[TM] = {0};
        float r1[TM] = {0};
        float inter[TN];
        float v0[TN];
        float v1[TN];
        const int thrN = simd_lid;
        const int simdM = simd_gid;
        int bm = simdM * TM;
        int bn = thrN * TN;
        int out_row = tid_x * blockM + bm;
        if (out_row >= N) return;
        out_row = out_row + TM <= N ? out_row : N - TM;
        const device T* mat = w + (size_t)out_row * (size_t)K;
        const int n_iter = K / blockN;
        for (int i = 0; i < n_iter; ++i) {
            for (int tn = 0; tn < TN; tn++) { v0[tn] = x[bn + tn]; }
            for (int tn = 0; tn < TN; tn++) { v1[tn] = x[(size_t)K + bn + tn]; }
            int mat_offset = 0;
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) { inter[tn] = static_cast<float>(mat[mat_offset + bn + tn]); }
                for (int tn = 0; tn < TN; tn++) { r0[tm] += inter[tn] * v0[tn]; }
                for (int tn = 0; tn < TN; tn++) { r1[tm] += inter[tn] * v1[tn]; }
                mat_offset += K;
            }
            bn += blockN;
        }
        for (int tm = 0; tm < TM; tm++) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                r0[tm] += simd_shuffle_down(r0[tm], sn);
                r1[tm] += simd_shuffle_down(r1[tm], sn);
            }
        }
        if (simd_lid == 0) {
            for (int tm = 0; tm < TM; tm++) { out[out_row + tm] = r0[tm]; out[(size_t)N + out_row + tm] = r1[tm]; }
        }
        """

    nonisolated(unsafe) static let router2RowKernel = MLXFast.metalKernel(
        name: "track_router_gemv_2row",
        inputNames: ["x", "w"],
        outputNames: ["out"],
        source: router2RowSource, ensureRowContiguous: true)

    /// x float32 [2, K]; w bf16 [N, K] -> float32 [2, N] logits, row r bit-identical
    /// to `track_router_gemv` on row r alone.
    static func router2Row(x: MLXArray, w: MLXArray) -> MLXArray {
        let K = w.dim(1), N = w.dim(0)
        precondition(x.dtype == .float32 && x.size == 2 * K)
        precondition(w.dtype == .bfloat16 && K % 128 == 0 && N % 16 == 0)
        let rps = K == 2560 && N == 512 ? 1 : 4
        return router2RowKernel(
            [x.reshaped(2, K), w],
            template: [("T", w.dtype), ("K", K), ("N", N), ("RPS", rps)],
            grid: (32 * (N / (4 * rps)), 1, 4), threadGroup: (32, 1, 4),
            outputShapes: [[2, N]], outputDTypes: [.float32])[0]
    }

    /// Kill switch. ON by default; a marker file turns it OFF, so one build
    /// serves both arms of a local A/B (the worker is a separate process and does
    /// not inherit the launching shell's environment, so this cannot be an env
    /// var). Nothing creates the file: a measured box has no `/tmp/mlx-v2`.
    nonisolated(unsafe) static let routerTwoRowEnabled: Bool =
        !FileManager.default.fileExists(atPath: "/tmp/mlx-v2/no-router2row")

    // MARK: 2. HC mixer down+inject -- two rows, one quantized weight walk.
    //
    // The VPT == 1 body of `track_mixer_down_inject_1`, twice over, sharing the
    // walk. `RPS` is the tip's shipped one-token ownership (1 row/simdgroup).

    static let downInject2RowSource = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        constexpr int RPS = \(TrackFastMixerKernels.downRowsPerSimdgroup);
        constexpr int NT = ND / (2 * RPS);
        const device T* n0 = normed;
        const device T* n1 = normed + (size_t)KD;
        if (tile < NT) {
            float r0[RPS];
            float r1[RPS];
            qmv_fast_reg2<T, GS, BITS, RPS>(wd, sd, bd, n0, n1, KD, tile * (2 * RPS) + (int)sg * RPS, lid, r0, r1);
            if (lid == 0) {
                for (int i = 0; i < RPS; ++i) {
                    const int c = tile * (2 * RPS) + (int)sg * RPS + i;
                    const T l0 = static_cast<T>(r0[i]);
                    const T l1 = static_cast<T>(r1[i]);
                    lo[c] = l0; act[c] = mlx_silu(l0);
                    lo[ND + c] = l1; act[ND + c] = mlx_silu(l1);
                }
            }
        } else if (HAS_INJECT) {
            static_assert(HC == 4, "one-row inject tiles require four HC rows");
            const int row = (tile - NT) * 2 + (int)sg;
            switch (row) {
                case 0: track_inject_qmv_row2<T, GS, BITS, KD, 0>(wi, si, bi, n0, n1, inj, HC, lid); break;
                case 1: track_inject_qmv_row2<T, GS, BITS, KD, 1>(wi, si, bi, n0, n1, inj, HC, lid); break;
                case 2: track_inject_qmv_row2<T, GS, BITS, KD, 2>(wi, si, bi, n0, n1, inj, HC, lid); break;
                case 3: track_inject_qmv_row2<T, GS, BITS, KD, 3>(wi, si, bi, n0, n1, inj, HC, lid); break;
            }
        }
        """

    static let header2 =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
        + TrackFastMoEKernels.regHelpers + TrackFastKernels.mixerHeadHeaderTail
        + TrackFastMoEKernels.wideDecls + twoRowHelpers

    nonisolated(unsafe) static let downInject2RowKernel = MLXFast.metalKernel(
        name: "track_mixer_down_inject_2row",
        inputNames: ["normed", "wd", "sd", "bd", "wi", "si", "bi"],
        outputNames: ["lo", "act", "inj"],
        source: downInject2RowSource, header: header2, ensureRowContiguous: true)

    /// normed [2, KD] -> lo/act [2, ND], inj [2, HC]; row r bit-identical to the
    /// serial `track_mixer_down_inject_1` on row r alone.
    static func downInject2Row(
        normed: MLXArray, down: TrackQuantWeight, inject: TrackQuantWeight?
    ) -> (lo: MLXArray, act: MLXArray, inj: MLXArray) {
        let KD = normed.dim(-1), ND = down.rows
        let HC = inject?.rows ?? 4
        precondition(normed.dim(0) == 2 && ND % 8 == 0 && KD % 512 == 0 && down.bits == 4)
        let inj = inject ?? down
        let rps = TrackFastMixerKernels.downRowsPerSimdgroup
        let tiles = ND / (2 * rps) + (inject != nil ? 2 : 0)
        let o = downInject2RowKernel(
            [normed, down.weight, down.scales, down.biases!,
             inj.weight, inj.scales, inj.biases!],
            template: [
                ("T", normed.dtype), ("GS", down.groupSize), ("BITS", down.bits), ("KD", KD),
                ("ND", ND), ("HC", HC), ("HAS_INJECT", inject != nil),
            ],
            grid: (32, tiles * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[2, ND], [2, ND], [2, HC]],
            outputDTypes: [normed.dtype, normed.dtype, normed.dtype])
        return (lo: o[0], act: o[1], inj: o[2])
    }

    /// Kill switch, ON by default (see `routerTwoRowEnabled`).
    nonisolated(unsafe) static let hcDownTwoRowEnabled: Bool =
        !FileManager.default.fileExists(atPath: "/tmp/mlx-v2/no-hcdown2row")

    // MARK: 3. dense quantized GEMV -- two rows, one quantized weight walk.
    //
    // `affine_qmv_fast`'s ownership (4 output rows per simdgroup, two simdgroups
    // per threadgroup) with two activation vectors.

    static let dense2RowSource = """
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        const int tile = (int)threadgroup_position_in_grid.y;
        const int out_row = (tile * 2 + (int)sg) * 4;
        if (out_row >= N) return;
        float r0[4];
        float r1[4];
        qmv_fast_reg2<T, GS, BITS, 4>(w, scales, biases, x0, x1, K, out_row, lid, r0, r1);
        if (lid == 0) {
            for (int i = 0; i < 4; ++i) {
                y0[out_row + i] = static_cast<T>(r0[i]);
                y1[out_row + i] = static_cast<T>(r1[i]);
            }
        }
        """

    static let headerDense =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
        + TrackFastMoEKernels.regHelpers + twoRowHelpers

    nonisolated(unsafe) static let dense2RowKernel = MLXFast.metalKernel(
        name: "track_dense_qmv_2row",
        inputNames: ["x0", "x1", "w", "scales", "biases"],
        outputNames: ["y0", "y1"],
        source: dense2RowSource, header: headerDense, ensureRowContiguous: true)

    /// x0, x1 [K] in the weight dtype; q a 4-bit affine [N, K] -> two [N] rows.
    static func dense2Row(x0: MLXArray, x1: MLXArray, q: TrackQuantWeight) -> (MLXArray, MLXArray) {
        let K = x0.size, N = q.rows
        precondition(q.bits == 4 && N % 8 == 0 && K % 512 == 0)
        let outs = dense2RowKernel(
            [x0.reshaped(K), x1.reshaped(K), q.weight, q.scales, q.biases!],
            template: [("T", x0.dtype), ("GS", q.groupSize), ("BITS", q.bits), ("K", K), ("N", N)],
            grid: (32, N / 4, 1), threadGroup: (32, 2, 1),
            outputShapes: [[N], [N]], outputDTypes: [x0.dtype, x0.dtype])
        return (outs[0], outs[1])
    }
}
