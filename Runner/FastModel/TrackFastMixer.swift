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
    /// PB-124 rows-per-simdgroup. Default ON keeps the live MIX2ROW count and
    /// the two-tile inject split. `TRACK_RPS=0` restores original ownership
    /// (down 4 rows/sg; one two-simdgroup inject tile).
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_RPS"] ?? "1") != "0"
    }()

    static let header =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + TrackFastMoEKernels.regHelpers
        + TrackFastKernels.mixerHeadHeaderTail + TrackFastMoEKernels.wideHelpers
    /// One-token instantiations: the wide bodies are replaced by their declarations.
    static let header1 =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + TrackFastMoEKernels.regHelpers
        + TrackFastKernels.mixerHeadHeaderTail + TrackFastMoEKernels.wideDecls

    /// One generator for the generic runtime-bound source and the staged
    /// source-string variants. `kd` / `hc` / `window` nil keeps the kernel
    /// template identifiers; a value interpolates that bound as a literal.
    /// RPS and INJ_SPLIT stay kernel templates so TRACK_RPS can restore the
    /// original ownership without a second source string.
    static func makeDownInjectSource(kd: Int? = nil, hc: Int? = nil, window: Int? = nil) -> String {
        let vpt = window.map(String.init) ?? "VPT"
        let kdTok = kd.map(String.init) ?? "KD"
        let hcTok = hc.map(String.init) ?? "HC"
        let stagedK = kd != nil
        let fastCall = stagedK
            ? "qmv_fast_reg<T, GS, BITS, \(kdTok), RPS>(wd, sd, bd, normed, tile * (2 * RPS) + (int)sg * RPS, lid, r)"
            : "qmv_fast_reg<T, GS, BITS, RPS>(wd, sd, bd, normed, KD, tile * (2 * RPS) + (int)sg * RPS, lid, r)"
        let wideFull = stagedK
            ? "qmv_wide_reg_full<T, GS, BITS, \(vpt), 8, false, \(kdTok)>(wd, sd, bd, normed, KD, \(vpt), row, lid, r)"
            : "qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wd, sd, bd, normed, KD, VPT, row, lid, r)"
        let widePart = stagedK
            ? "qmv_wide_reg_partial<T, GS, BITS, \(vpt), 8, \(kdTok), \(hcTok)>(wi, si, bi, normed, KD, HC, \(vpt), fp, sg, lid, r, valid, row)"
            : "qmv_wide_reg_partial<T, GS, BITS, VPT, 8>(wi, si, bi, normed, KD, HC, VPT, fp, sg, lid, r, valid, row)"
        func injRow(_ n: Int) -> String {
            "track_inject_qmv_row<T, GS, BITS, \(kdTok), \(n), EXACT_TAIL>(wi, si, bi, normed, inj, lid)"
        }
        return """
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        // MLXFAST-MIX2ROW: RPS is a template. S>1 and TRACK_RPS=0 pass 4.
        static_assert(RPS == 1 || RPS == 2 || RPS == 4, "down row ownership");
        constexpr int NT = ND / (2 * RPS);
        if (tile < NT) {
            if constexpr (\(vpt) == 1) {
                float r[RPS];
                \(fastCall);
                if (lid == 0) {
                    for (int i = 0; i < RPS; ++i) {
                        const T l = static_cast<T>(r[i]);
                        lo[tile * (2 * RPS) + (int)sg * RPS + i] = l;
                        act[tile * (2 * RPS) + (int)sg * RPS + i] = mlx_silu(l);
                    }
                }
            } else {
                float r[\(vpt)];
                const int row = tile * 8 + (int)sg * 4 + (int)(lid / 8);
                \(wideFull);
                if ((lid % 8) == 0) {
                    for (int v = 0; v < \(vpt); ++v) {
                        const T l = static_cast<T>(r[v]);
                        lo[v * ND + row] = l;
                        act[v * ND + row] = mlx_silu(l);
                    }
                }
            }
        } else if (HAS_INJECT) {
            if constexpr (\(vpt) == 1) {
                if constexpr (INJ_SPLIT) {
                    // MLXFAST-INJSPLIT: only ownership changes; each row keeps its K walk.
                    static_assert(\(hcTok) == 4, "one-row inject tiles require four HC rows");
                    const int row = (tile - NT) * 2 + (int)sg;
                    switch (row) {
                        case 0: \(injRow(0)); break;
                        case 1: \(injRow(1)); break;
                        case 2: \(injRow(2)); break;
                        case 3: \(injRow(3)); break;
                    }
                } else {
                    track_inject_qmv<T, GS, BITS, \(kdTok), \(hcTok), 4, EXACT_TAIL>(wi, si, bi, normed, inj, sg, lid);
                }
            } else {
                threadgroup float fp[8 * \(vpt)];
                float r[\(vpt)];
                bool valid = false; int row = 0;
                \(widePart);
                if (valid) {
                    for (int v = 0; v < \(vpt); ++v) { inj[v * HC + row] = static_cast<T>(r[v]); }
                }
            }
        }
        """
    }

    /// Generic runtime-bound source. TRACK_HC_STAGING=0 dispatches this.
    static let downInjectSource = makeDownInjectSource()

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

    /// Staged source-string variants: one per live window, live K and HC baked in.
    nonisolated(unsafe) static let downInjectStaged: [Int: MLXFast.MLXFastKernel] = {
        var d: [Int: MLXFast.MLXFastKernel] = [:]
        for s in TrackHCStaging.liveWindows {
            d[s] = MLXFast.metalKernel(
                name: "track_mixer_down_inject_k\(TrackHCStaging.liveHidden)_hc\(TrackHCStaging.liveHC)_s\(s)",
                inputNames: ["normed", "wd", "sd", "bd", "wi", "si", "bi"],
                outputNames: ["lo", "act", "inj"],
                source: makeDownInjectSource(
                    kd: TrackHCStaging.liveHidden, hc: TrackHCStaging.liveHC, window: s),
                header: s == 1 ? header1 : header, ensureRowContiguous: true)
        }
        return d
    }()

    static func downInject(
        normed: MLXArray, down: TrackQuantWeight, inject: TrackQuantWeight?,
        staging: Bool = TrackHCStaging.enabled
    ) -> (lo: MLXArray, act: MLXArray, inj: MLXArray) {
        let S = normed.dim(0), KD = normed.dim(1), ND = down.rows
        let HC = inject?.rows ?? 4
        precondition(S >= 1 && S <= 8 && ND % 8 == 0 && KD % 512 == 0 && down.bits == 4)
        let inj = inject ?? down
        // Per-site fallback: down RPS and inject split each restore their own
        // original ownership when TRACK_RPS=0. Launch size stays 64.
        let rowsPerSimdgroup = S == 1 && enabled ? downRowsPerSimdgroup : 4
        let injSplit = S == 1 && enabled
        let tiles = ND / (2 * rowsPerSimdgroup) + (inject != nil ? (injSplit ? 2 : 1) : 0)
        let kernel: MLXFast.MLXFastKernel
        if TrackHCStaging.mixerPath(s: S, kd: KD, hc: HC, staging: staging) == .staged,
            let staged = downInjectStaged[S]
        {
            kernel = staged
        } else {
            kernel = S == 1 ? downInjectKernel1 : downInjectKernel
        }
        let outs = kernel(
            [normed, down.weight, down.scales, down.biases!, inj.weight, inj.scales, inj.biases!],
            template: [
                ("T", normed.dtype), ("GS", down.groupSize), ("BITS", down.bits), ("KD", KD), ("ND", ND),
                ("HC", HC), ("VPT", S), ("HAS_INJECT", inject != nil),
                ("RPS", rowsPerSimdgroup), ("INJ_SPLIT", injSplit),
                ("EXACT_TAIL", TrackFastMoEKernels.useExactTail(k: KD, bits: down.bits)),
            ],
            grid: (32, tiles * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[S, ND], [S, ND], [S, HC]], outputDTypes: [normed.dtype, normed.dtype, normed.dtype])
        return (outs[0], outs[1], outs[2])
    }

    /// lo [S, LW] (pre-silu), normed [S, HC*H], inj [S, HC] -> input [S, H], inject [S, HC].
    /// Tile t owns columns 2t, 2t+1 across the HC streams (8 rows); slot s ->
    /// row (2t + (s & 1)) + H * (s >> 1). grid threads (32, 2 * H/2, 1), tg (32, 2, 1).
    static func makeUpMixSource(lw: Int? = nil, hc: Int? = nil, window: Int? = nil) -> String {
        let vpt = window.map(String.init) ?? "VPT"
        let lwTok = lw.map(String.init) ?? "LW"
        let hcTok = hc.map(String.init) ?? "HC"
        let stagedK = lw != nil
        let rowsCall = stagedK
            ? "qmv_reg_rows<T, GS, BITS, \(lwTok), false, EXACT_TAIL>(wu, su, bu, act, rows, lid, r)"
            : "qmv_reg_rows<T, GS, BITS, false, EXACT_TAIL>(wu, su, bu, act, LW, rows, lid, r)"
        let wideFull = stagedK
            ? "qmv_wide_reg_full<T, GS, BITS, \(vpt), 8, false, \(lwTok)>(wu, su, bu, act, LW, \(vpt), row, lid, r)"
            : "qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wu, su, bu, act, LW, VPT, row, lid, r)"
        return """
        const int tile = (int)threadgroup_position_in_grid.y;
        const int d0 = 2 * tile;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup float res[8][\(vpt)];
        if constexpr (\(vpt) == 1) {
            int rows[4];
            for (int i = 0; i < 4; ++i) { const int s = (int)sg * 4 + i; rows[i] = d0 + (s & 1) + H * (s >> 1); }
            float r[4];
            \(rowsCall);
            if (lid == 0) { for (int i = 0; i < 4; ++i) { res[(int)sg * 4 + i][0] = r[i]; } }
        } else {
            const int s = (int)sg * 4 + (int)(lid / 8);
            const int row = d0 + (s & 1) + H * (s >> 1);
            float r[\(vpt)];
            \(wideFull);
            if ((lid % 8) == 0) { for (int v = 0; v < \(vpt); ++v) { res[s][v] = r[v]; } }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = sg * 32 + lid;
        if (t < 2) {
            const int d = d0 + (int)t;
            for (int v = 0; v < \(vpt); ++v) {
                T acc = T(0);
                for (int s = 0; s < \(hcTok); ++s) {
                    const T w = static_cast<T>(res[s * 2 + (int)t][v]);
                    const T sgm = mlx_sigmoid(w);
                    const T p = sgm * normed[(size_t)v * (size_t)(HC * H) + (size_t)(s * H + d)];
                    acc = acc + p;
                }
                input[(size_t)v * (size_t)H + (size_t)d] = acc;
                if (EMIT_F32) { inputF[(size_t)v * (size_t)H + (size_t)d] = static_cast<float>(acc); }
            }
        }
        if (HAS_INJECT && tile == 0 && t < (uint)(HC * \(vpt))) {
            const T x = inj[t];
            inject[t] = T(2) * mlx_sigmoid(x);
        }
        """
    }

    static let upMixSource = makeUpMixSource()

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

    nonisolated(unsafe) static let upMixStaged: [Int: MLXFast.MLXFastKernel] = {
        var d: [Int: MLXFast.MLXFastKernel] = [:]
        for s in TrackHCStaging.liveWindows {
            d[s] = MLXFast.metalKernel(
                name: "track_mixer_up_mix_lw\(TrackHCStaging.liveLowrank)_hc\(TrackHCStaging.liveHC)_s\(s)",
                inputNames: ["act", "normed", "wu", "su", "bu", "inj"],
                outputNames: ["input", "inject", "inputF"],
                source: makeUpMixSource(
                    lw: TrackHCStaging.liveLowrank, hc: TrackHCStaging.liveHC, window: s),
                header: s == 1 ? header1 : header, ensureRowContiguous: true)
        }
        return d
    }()

    static func upMix(
        act: MLXArray, normed: MLXArray, up: TrackQuantWeight, inj: MLXArray, hcCount: Int, hidden: Int,
        hasInject: Bool, emitF32: Bool = false, staging: Bool = TrackHCStaging.enabled
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray) {
        let S = act.dim(0), LW = act.dim(1)
        precondition(S >= 1 && S <= 8 && hidden % 2 == 0 && up.rows == hcCount * hidden && up.bits == 4)
        precondition(LW % 32 == 0 && LW < 512 + 256)  // K = 320: one full block + a tail, the `qmv` normal branch
        let kernel: MLXFast.MLXFastKernel
        if TrackHCStaging.upMixPath(s: S, lw: LW, hc: hcCount, staging: staging) == .staged,
            let staged = upMixStaged[S]
        {
            kernel = staged
        } else {
            kernel = S == 1 ? upMixKernel1 : upMixKernel
        }
        let outs = kernel(
            [act, normed, up.weight, up.scales, up.biases!, inj],
            template: [
                ("T", act.dtype), ("GS", up.groupSize), ("BITS", up.bits), ("H", hidden), ("HC", hcCount),
                ("LW", LW), ("VPT", S), ("HAS_INJECT", hasInject), ("EMIT_F32", emitF32),
                ("EXACT_TAIL", TrackFastMoEKernels.useExactTail(k: LW, bits: up.bits)),
            ],
            grid: (32, (hidden / 2) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[S, hidden], [S, hcCount], [emitF32 ? S : 1, emitF32 ? hidden : 1]],
            outputDTypes: [act.dtype, act.dtype, .float32])
        return (outs[0], outs[1], outs[2])
    }
}
