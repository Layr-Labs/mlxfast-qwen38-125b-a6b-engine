// Hyper-connection mixer in three launches for decode windows (S <= 8):
//   injectNorm  ->  [down GEMV || inject GEMV, silu]  ->  [up GEMV + combine]
// Every GEMV walk is MLX's own for the window size: one token routes the
// 320-row down to `qmv_fast`, the 4-row inject to `qmv`'s small-N branch and
// the K=320 up to `qmv`'s normal branch; two to eight tokens route all three
// to `qmv_wide` (full tiles for down/up, the short-tile fold for inject). The
// replicas keep the lanes, the accumulation order and the reductions; only
// where the results land changes.

import Foundation
import MLX
import MLXFast

enum TrackFastMixerKernels {
    // MLXFAST-MIX2ROW: source-time choice for S == 1 only; use 1 for the
    // optional traffic/parallelism experiment, or 4 for the original ownership.
    static let downRowsPerSimdgroup = 1

    static let header =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + TrackFastMoEKernels.regHelpers
        + TrackFastKernels.mixerHeadHeaderTail + TrackFastMoEKernels.wideHelpers
    /// One-token instantiations: the wide bodies are replaced by their declarations.
    static let header1 =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + TrackFastMoEKernels.regHelpers
        + TrackFastKernels.mixerHeadHeaderTail + TrackFastMoEKernels.wideDecls

    /// normed [S, KD] -> lo [S, ND] (down), inj [S, HC] (inject).
    /// S == 1: each simdgroup owns downRowsPerSimdgroup adjacent down rows.
    /// S == 1: two inject tiles each own two rows, one per simdgroup.
    /// S > 1 retains 8 down rows/tile and one two-simdgroup inject tile.
    /// grid threads (32, 2 * tiles, 1), tg (32, 2, 1).
    static let downInjectSource = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        // MLXFAST-MIX2ROW: the wide path retains its original four-row ownership.
        constexpr int RPS = VPT == 1 ? \(downRowsPerSimdgroup) : 4;
        static_assert(RPS == 1 || RPS == 2 || RPS == 4, "down row ownership");
        constexpr int NT = ND / (2 * RPS);
        if (tile < NT) {
            if constexpr (VPT == 1) {
                float r[RPS];
                qmv_fast_reg<T, GS, BITS, RPS>(wd, sd, bd, normed, KD, tile * (2 * RPS) + (int)sg * RPS, lid, r);
                if (lid == 0) {
                    for (int i = 0; i < RPS; ++i) {
                        const T l = static_cast<T>(r[i]);
                        lo[tile * (2 * RPS) + (int)sg * RPS + i] = l;
                        act[tile * (2 * RPS) + (int)sg * RPS + i] = mlx_silu(l);
                    }
                }
            } else {
                float r[VPT];
                const int row = tile * 8 + (int)sg * 4 + (int)(lid / 8);
                qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wd, sd, bd, normed, KD, VPT, row, lid, r);
                if ((lid % 8) == 0) {
                    for (int v = 0; v < VPT; ++v) {
                        const T l = static_cast<T>(r[v]);
                        lo[v * ND + row] = l;
                        act[v * ND + row] = mlx_silu(l);
                    }
                }
            }
        } else if (HAS_INJECT) {
            if constexpr (VPT == 1) {
                // MLXFAST-INJSPLIT: only ownership changes; each row keeps its K walk.
                static_assert(HC == 4, "one-row inject tiles require four HC rows");
                const int row = (tile - NT) * 2 + (int)sg;
                switch (row) {
                    case 0: track_inject_qmv_row<T, GS, BITS, KD, 0>(wi, si, bi, normed, inj, lid); break;
                    case 1: track_inject_qmv_row<T, GS, BITS, KD, 1>(wi, si, bi, normed, inj, lid); break;
                    case 2: track_inject_qmv_row<T, GS, BITS, KD, 2>(wi, si, bi, normed, inj, lid); break;
                    case 3: track_inject_qmv_row<T, GS, BITS, KD, 3>(wi, si, bi, normed, inj, lid); break;
                }
            } else {
                threadgroup float fp[8 * VPT];
                float r[VPT];
                bool valid = false; int row = 0;
                qmv_wide_reg_partial<T, GS, BITS, VPT, 8>(wi, si, bi, normed, KD, HC, VPT, fp, sg, lid, r, valid, row);
                if (valid) {
                    for (int v = 0; v < VPT; ++v) { inj[v * HC + row] = static_cast<T>(r[v]); }
                }
            }
        }
        """

    nonisolated(unsafe) static let downInjectKernel = MLXFast.metalKernel(
        name: "track_mixer_down_inject",
        inputNames: ["normed", "wd", "sd", "bd", "wi", "si", "bi"],
        outputNames: ["lo", "act", "inj"],
        source: downInjectSource, header: header, ensureRowContiguous: true)
    nonisolated(unsafe) static let downInjectKernel1 = MLXFast.metalKernel(
        name: "track_mixer_down_inject_1",
        inputNames: ["normed", "wd", "sd", "bd", "wi", "si", "bi"],
        outputNames: ["lo", "act", "inj"],
        source: downInjectSource, header: header1, ensureRowContiguous: true)

    static func downInject(
        normed: MLXArray, down: TrackQuantWeight, inject: TrackQuantWeight?
    ) -> (lo: MLXArray, act: MLXArray, inj: MLXArray) {
        let S = normed.dim(0), KD = normed.dim(1), ND = down.rows
        let HC = inject?.rows ?? 4
        precondition(S >= 1 && S <= 8 && ND % 8 == 0 && KD % 512 == 0 && down.bits == 4)
        let inj = inject ?? down
        // MLXFAST-MIX2ROW: match the source-time row count; launch size stays 64.
        let rowsPerSimdgroup = S == 1 ? downRowsPerSimdgroup : 4
        // MLXFAST-INJSPLIT: add two inject tiles only to the one-token path.
        let tiles = ND / (2 * rowsPerSimdgroup) + (inject != nil ? (S == 1 ? 2 : 1) : 0)
        let outs = (S == 1 ? downInjectKernel1 : downInjectKernel)(
            [normed, down.weight, down.scales, down.biases!, inj.weight, inj.scales, inj.biases!],
            template: [
                ("T", normed.dtype), ("GS", down.groupSize), ("BITS", down.bits), ("KD", KD), ("ND", ND),
                ("HC", HC), ("VPT", S), ("HAS_INJECT", inject != nil),
            ],
            grid: (32, tiles * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[S, ND], [S, ND], [S, HC]], outputDTypes: [normed.dtype, normed.dtype, normed.dtype])
        return (outs[0], outs[1], outs[2])
    }

    /// lo [S, LW] (pre-silu), normed [S, HC*H], inj [S, HC] -> input [S, H], inject [S, HC].
    /// Tile t owns columns 2t, 2t+1 across the HC streams (8 rows); slot s ->
    /// row (2t + (s & 1)) + H * (s >> 1). grid threads (32, 2 * H/2, 1), tg (32, 2, 1).
    static let upMixSource = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const int d0 = 2 * tile;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup float res[8][VPT];
        if constexpr (VPT == 1) {
            int rows[4];
            for (int i = 0; i < 4; ++i) { const int s = (int)sg * 4 + i; rows[i] = d0 + (s & 1) + H * (s >> 1); }
            float r[4];
            qmv_reg_rows<T, GS, BITS, false, (LW % get_pack_factor<BITS, 32>()) == 0>(wu, su, bu, act, LW, rows, lid, r);
            if (lid == 0) { for (int i = 0; i < 4; ++i) { res[(int)sg * 4 + i][0] = r[i]; } }
        } else {
            const int s = (int)sg * 4 + (int)(lid / 8);
            const int row = d0 + (s & 1) + H * (s >> 1);
            float r[VPT];
            qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wu, su, bu, act, LW, VPT, row, lid, r);
            if ((lid % 8) == 0) { for (int v = 0; v < VPT; ++v) { res[s][v] = r[v]; } }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = sg * 32 + lid;
        if (t < 2) {
            const int d = d0 + (int)t;
            for (int v = 0; v < VPT; ++v) {
                T acc = T(0);
                for (int s = 0; s < HC; ++s) {
                    const T w = static_cast<T>(res[s * 2 + (int)t][v]);
                    const T sgm = mlx_sigmoid(w);
                    const T p = sgm * normed[(size_t)v * (size_t)(HC * H) + (size_t)(s * H + d)];
                    acc = acc + p;
                }
                input[(size_t)v * (size_t)H + (size_t)d] = acc;
                if (EMIT_F32) { inputF[(size_t)v * (size_t)H + (size_t)d] = static_cast<float>(acc); }
            }
        }
        if (HAS_INJECT && tile == 0 && t < (uint)(HC * VPT)) {
            const T x = inj[t];
            inject[t] = T(2) * mlx_sigmoid(x);
        }
        """

    nonisolated(unsafe) static let upMixKernel = MLXFast.metalKernel(
        name: "track_mixer_up_mix",
        inputNames: ["act", "normed", "wu", "su", "bu", "inj"],
        outputNames: ["input", "inject", "inputF"],
        source: upMixSource, header: header, ensureRowContiguous: true)
    nonisolated(unsafe) static let upMixKernel1 = MLXFast.metalKernel(
        name: "track_mixer_up_mix_1",
        inputNames: ["act", "normed", "wu", "su", "bu", "inj"],
        outputNames: ["input", "inject", "inputF"],
        source: upMixSource, header: header1, ensureRowContiguous: true)

    static func upMix(
        act: MLXArray, normed: MLXArray, up: TrackQuantWeight, inj: MLXArray, hcCount: Int, hidden: Int,
        hasInject: Bool, emitF32: Bool = false
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray) {
        let S = act.dim(0), LW = act.dim(1)
        precondition(S >= 1 && S <= 8 && hidden % 2 == 0 && up.rows == hcCount * hidden && up.bits == 4)
        precondition(LW % 32 == 0 && LW < 512 + 256)  // K = 320: one full block + a tail, the `qmv` normal branch
        let outs = (S == 1 ? upMixKernel1 : upMixKernel)(
            [act, normed, up.weight, up.scales, up.biases!, inj],
            template: [
                ("T", act.dtype), ("GS", up.groupSize), ("BITS", up.bits), ("H", hidden), ("HC", hcCount),
                ("LW", LW), ("VPT", S), ("HAS_INJECT", hasInject), ("EMIT_F32", emitF32),
            ],
            grid: (32, (hidden / 2) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[S, hidden], [S, hcCount], [emitF32 ? S : 1, emitF32 ? hidden : 1]],
            outputDTypes: [act.dtype, act.dtype, .float32])
        return (outs[0], outs[1], outs[2])
    }
}
