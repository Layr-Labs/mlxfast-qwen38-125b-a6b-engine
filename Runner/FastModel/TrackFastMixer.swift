// Hyper-connection mixer for decode windows (S <= 8).
// `injectNorm` stays its own launch: it reduces across the block.
// `downUp` is the scored mixer's other launch. It runs the down walk and the
// inject walk, keeps `act` and `inj` in the threadgroup, then runs the up GEMV
// and the combine. The 320-row down walk and the 4-row inject walk stay
// separate. One token still uses split-K when that path is on, otherwise
// `qmv_fast` for the down rows and `qmv`'s small-N inject; the up rows stay on
// `qmv`. Two to eight tokens stay on `qmv_wide` (full tiles for down and up,
// the short-tile fold for inject). Lanes, accumulation, and `simd_sum` are the
// ones those walks already use.
// `downInject` and `upMix` remain the separate launches for the other callers.
// The fused launch is one threadgroup: `act` is the whole low-rank vector, and
// threadgroup memory does not survive across groups. Repeating the down walk
// inside every up tile would multiply that GEMV.

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
        if S == 1 && down.groupSize == 32 && (inject == nil || inject!.groupSize == 32)
            && TrackFastMixerSplitK.split > 0 {
            let o = TrackFastMixerSplitK.apply(normed, down: down, inject: inject)
            return (o[0], o[1], o[2])
        }
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
        auto sigmoid = [&](T value) -> T {
            if constexpr (metal::is_same_v<T, bfloat16_t>) {
                return sigmoid_lut[as_type<ushort>(static_cast<bfloat16_t>(value))];
            } else {
                return mlx_sigmoid(value);
            }
        };
        const int tile = (int)threadgroup_position_in_grid.y;
        const int d0 = 2 * tile;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup T products[8][VPT];
        if constexpr (VPT == 1) {
            float r[4];
            if constexpr (PACKED_ROWS) {
                qmv_reg<T, GS, BITS, (LW % get_pack_factor<BITS, 32>()) == 0>(wu, su, bu, act, LW, tile * 8 + (int)sg * 4, lid, r);
            } else {
                int rows[4];
                for (int i = 0; i < 4; ++i) { const int s = (int)sg * 4 + i; rows[i] = d0 + (s & 1) + H * (s >> 1); }
                qmv_reg_rows<T, GS, BITS, false, (LW % get_pack_factor<BITS, 32>()) == 0>(wu, su, bu, act, LW, rows, lid, r);
            }
            if (lid < 4 && (int)(sg * 2 + lid / 2) < HC) {
                const int slot = (int)sg * 4 + (int)lid;
                const float low = metal::select(r[0], r[1], (lid & 1u) != 0);
                const float high = metal::select(r[2], r[3], (lid & 1u) != 0);
                const T weight = static_cast<T>(metal::select(low, high, (lid & 2u) != 0));
                const int row = d0 + (slot & 1) + H * (slot >> 1);
                products[slot][0] = sigmoid(weight) * normed[row];
            }
        } else {
            const int s = (int)sg * 4 + (int)(lid / 8);
            const int row = d0 + (s & 1) + H * (s >> 1);
            float r[VPT];
            qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wu, su, bu, act, LW, VPT, row, lid, r);
            if ((lid % 8) == 0 && (s >> 1) < HC) {
                for (int v = 0; v < VPT; ++v) {
                    const T weight = static_cast<T>(r[v]);
                    products[s][v] = sigmoid(weight) * normed[(size_t)v * (size_t)(HC * H) + (size_t)row];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = sg * 32 + lid;
        if (t < 2) {
            const int d = d0 + (int)t;
            for (int v = 0; v < VPT; ++v) {
                T acc = T(0);
                for (int s = 0; s < HC; ++s) {
                    const T p = products[s * 2 + (int)t][v];
                    acc = acc + p;
                }
                input[(size_t)v * (size_t)H + (size_t)d] = acc;
                if (EMIT_F32) { inputF[(size_t)v * (size_t)H + (size_t)d] = static_cast<float>(acc); }
            }
        }
        if (HAS_INJECT && tile == 0 && t < (uint)(HC * VPT)) {
            const T x = inj[t];
            inject[t] = T(2) * sigmoid(x);
        }
        """

    nonisolated(unsafe) static let upMixKernel = MLXFast.metalKernel(
        name: "track_mixer_up_mix",
        inputNames: ["act", "normed", "wu", "su", "bu", "inj", "sigmoid_lut"],
        outputNames: ["input", "inject", "inputF"],
        source: upMixSource, header: header, ensureRowContiguous: true)
    nonisolated(unsafe) static let upMixKernel1 = MLXFast.metalKernel(
        name: "track_mixer_up_mix_1",
        inputNames: ["act", "normed", "wu", "su", "bu", "inj", "sigmoid_lut"],
        outputNames: ["input", "inject", "inputF"],
        source: upMixSource, header: header1, ensureRowContiguous: true)

    static func upMix(
        act: MLXArray, normed: MLXArray, up: TrackQuantWeight, inj: MLXArray, hcCount: Int, hidden: Int,
        hasInject: Bool, emitF32: Bool = false, packedRows: Bool = false
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray) {
        let S = act.dim(0), LW = act.dim(1)
        precondition(S >= 1 && S <= 8 && hidden % 2 == 0 && up.rows == hcCount * hidden && up.bits == 4)
        precondition(LW % 32 == 0 && LW < 512 + 256)  // K = 320: one full block + a tail, the `qmv` normal branch
        let sigmoidTable = act.dtype == .bfloat16 ? TrackBF16Functions.sigmoid : normed
        precondition(!packedRows || (S == 1 && hcCount == 4))
        let outs = (S == 1 ? upMixKernel1 : upMixKernel)(
            [act, normed, up.weight, up.scales, up.biases!, inj, sigmoidTable],
            template: [
                ("T", act.dtype), ("GS", up.groupSize), ("BITS", up.bits), ("H", hidden), ("HC", hcCount),
                ("LW", LW), ("VPT", S), ("HAS_INJECT", hasInject), ("EMIT_F32", emitF32),
                ("PACKED_ROWS", packedRows),
            ],
            grid: (32, (hidden / 2) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[S, hidden], [S, hcCount], [emitF32 ? S : 1, emitF32 ? hidden : 1]],
            outputDTypes: [act.dtype, act.dtype, .float32])
        return (outs[0], outs[1], outs[2])
    }

    /// Down walk, inject walk, then up GEMV, one launch. `act` and `inj` are
    /// staged in the threadgroup. The up walks are the existing `qmv` / `qmv_wide`
    /// calls, which take a device pointer, so the threadgroup values are published
    /// once into `act` and `inj` before those walks. One threadgroup covers every
    /// tile: a second group cannot see this group's `act`.
    static let downUpSource = """
        auto sigmoid = [&](T value) -> T {
            if constexpr (metal::is_same_v<T, bfloat16_t>) {
                return sigmoid_lut[as_type<ushort>(static_cast<bfloat16_t>(value))];
            } else {
                return mlx_sigmoid(value);
            }
        };
        static_assert(NSG >= 2, "up mix uses two simdgroups");
        static_assert(KD % 512 == 0 && ND % 8 == 0 && H % 2 == 0, "mixer shape");
        static_assert(VPT >= 1 && VPT <= 8, "decode window");
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        const uint linear = sg * 32u + lid;
        constexpr int kThreads = NSG * 32;
        constexpr int kDownScratch = KD / 512 * 2 * 32;
        constexpr int kInjScratch = KD / 256 * 32;
        constexpr int kScratch = (VPT == 1 && SPLIT > 0)
            ? (kDownScratch > kInjScratch ? kDownScratch : kInjScratch) : 1;
        threadgroup float scratch[kScratch];
        threadgroup T act_tg[VPT * ND];
        threadgroup T inj_tg[VPT * HC];
        threadgroup float fp[8 * VPT];
        threadgroup T products[8][VPT];
        if ((int)threadgroup_position_in_grid.y != 0 || (int)threadgroup_position_in_grid.z != 0) {
            return;
        }
        if constexpr (VPT == 1 && SPLIT > 0) {
            static_assert(SPLIT == NSG, "split-K simdgroups");
            constexpr int RPS = 2;
            constexpr int DN = ND / RPS;
            for (int tile = 0; tile < DN; ++tile) {
                float r[RPS];
                research_split_qmv<T, KD, 16, RPS, SPLIT, true>(
                    wd, sd, bd, normed, tile * RPS, sg, lid, scratch, r);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0 && lid == 0) {
                    for (int i = 0; i < RPS; ++i) {
                        const T l = static_cast<T>(r[i]);
                        lo[tile * RPS + i] = l;
                        act_tg[tile * RPS + i] = mlx_silu(l);
                    }
                }
            }
            if constexpr (HAS_INJECT) {
                for (int row = 0; row < HC; ++row) {
                    float r[1];
                    research_split_qmv<T, KD, 8, 1, SPLIT, true>(
                        wi, si, bi, normed, row, sg, lid, scratch, r);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg == 0 && lid == 0) { inj_tg[row] = static_cast<T>(r[0]); }
                }
            }
        } else if constexpr (VPT == 1) {
            static_assert(NSG == 2, "one-token down keeps two simdgroups");
            constexpr int RPS = \(downRowsPerSimdgroup);
            static_assert(RPS == 1 || RPS == 2 || RPS == 4, "down row ownership");
            constexpr int NT = ND / (2 * RPS);
            for (int tile = 0; tile < NT; ++tile) {
                float r[RPS];
                qmv_fast_reg<T, GS, BITS, RPS>(
                    wd, sd, bd, normed, KD, tile * (2 * RPS) + (int)sg * RPS, lid, r);
                if (lid == 0) {
                    for (int i = 0; i < RPS; ++i) {
                        const T l = static_cast<T>(r[i]);
                        const int at = tile * (2 * RPS) + (int)sg * RPS + i;
                        lo[at] = l;
                        act_tg[at] = mlx_silu(l);
                    }
                }
            }
            if constexpr (HAS_INJECT) {
                static_assert(HC == 4, "one-row inject tiles require four HC rows");
                for (int it = 0; it < 2; ++it) {
                    const int row = it * 2 + (int)sg;
                    switch (row) {
                        case 0: track_inject_qmv_row<T, GS, BITS, KD, 0>(wi, si, bi, normed, inj, lid); break;
                        case 1: track_inject_qmv_row<T, GS, BITS, KD, 1>(wi, si, bi, normed, inj, lid); break;
                        case 2: track_inject_qmv_row<T, GS, BITS, KD, 2>(wi, si, bi, normed, inj, lid); break;
                        case 3: track_inject_qmv_row<T, GS, BITS, KD, 3>(wi, si, bi, normed, inj, lid); break;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
                if (linear < (uint)HC) { inj_tg[linear] = inj[linear]; }
            }
        } else {
            static_assert(NSG == 2, "wide mixer keeps two simdgroups");
            constexpr int NT = ND / 8;
            for (int tile = 0; tile < NT; ++tile) {
                float r[VPT];
                const int row = tile * 8 + (int)sg * 4 + (int)(lid / 8);
                qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(
                    wd, sd, bd, normed, KD, VPT, row, lid, r);
                if ((lid % 8) == 0) {
                    for (int v = 0; v < VPT; ++v) {
                        const T l = static_cast<T>(r[v]);
                        lo[v * ND + row] = l;
                        act_tg[v * ND + row] = mlx_silu(l);
                    }
                }
            }
            if constexpr (HAS_INJECT) {
                float r[VPT];
                bool valid = false;
                int row = 0;
                qmv_wide_reg_partial<T, GS, BITS, VPT, 8>(
                    wi, si, bi, normed, KD, HC, VPT, fp, sg, lid, r, valid, row);
                if (valid) {
                    for (int v = 0; v < VPT; ++v) { inj_tg[v * HC + row] = static_cast<T>(r[v]); }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = (int)linear; i < VPT * ND; i += kThreads) { act[i] = act_tg[i]; }
        if constexpr (HAS_INJECT) {
            for (int i = (int)linear; i < VPT * HC; i += kThreads) { inj[i] = inj_tg[i]; }
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (int tile = 0; tile < H / 2; ++tile) {
            const int d0 = 2 * tile;
            if (sg < 2) {
                if constexpr (VPT == 1) {
                    float r[4];
                    if constexpr (PACKED_ROWS) {
                        qmv_reg<T, GSU, BITS, (ND % get_pack_factor<BITS, 32>()) == 0>(
                            wu, su, bu, act, ND, tile * 8 + (int)sg * 4, lid, r);
                    } else {
                        int rows[4];
                        for (int i = 0; i < 4; ++i) {
                            const int s = (int)sg * 4 + i;
                            rows[i] = d0 + (s & 1) + H * (s >> 1);
                        }
                        qmv_reg_rows<T, GSU, BITS, false, (ND % get_pack_factor<BITS, 32>()) == 0>(
                            wu, su, bu, act, ND, rows, lid, r);
                    }
                    if (lid < 4 && (int)(sg * 2 + lid / 2) < HC) {
                        const int slot = (int)sg * 4 + (int)lid;
                        const float low = metal::select(r[0], r[1], (lid & 1u) != 0);
                        const float high = metal::select(r[2], r[3], (lid & 1u) != 0);
                        const T weight = static_cast<T>(metal::select(low, high, (lid & 2u) != 0));
                        const int row = d0 + (slot & 1) + H * (slot >> 1);
                        products[slot][0] = sigmoid(weight) * normed[row];
                    }
                } else {
                    const int s = (int)sg * 4 + (int)(lid / 8);
                    const int row = d0 + (s & 1) + H * (s >> 1);
                    float r[VPT];
                    qmv_wide_reg_full<T, GSU, BITS, VPT, 8, false>(
                        wu, su, bu, act, ND, VPT, row, lid, r);
                    if ((lid % 8) == 0 && (s >> 1) < HC) {
                        for (int v = 0; v < VPT; ++v) {
                            const T weight = static_cast<T>(r[v]);
                            products[s][v] = sigmoid(weight) * normed[(size_t)v * (size_t)(HC * H) + (size_t)row];
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint t = linear;
            if (t < 2) {
                const int d = d0 + (int)t;
                for (int v = 0; v < VPT; ++v) {
                    T acc = T(0);
                    for (int s = 0; s < HC; ++s) {
                        const T p = products[s * 2 + (int)t][v];
                        acc = acc + p;
                    }
                    input[(size_t)v * (size_t)H + (size_t)d] = acc;
                    if (EMIT_F32) { inputF[(size_t)v * (size_t)H + (size_t)d] = static_cast<float>(acc); }
                }
            }
            if (HAS_INJECT && tile == 0 && t < (uint)(HC * VPT)) {
                const T x = inj[t];
                inject[t] = T(2) * sigmoid(x);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """

    static let downUpHeader = header + TrackFastMixerSplitK.helper
    static let downUpHeader1 = header1 + TrackFastMixerSplitK.helper

    nonisolated(unsafe) static let downUpKernel = MLXFast.metalKernel(
        name: "track_mixer_down_up",
        inputNames: ["normed", "wd", "sd", "bd", "wi", "si", "bi", "wu", "su", "bu", "sigmoid_lut"],
        outputNames: ["lo", "inj", "act", "input", "inject", "inputF"],
        source: downUpSource, header: downUpHeader, ensureRowContiguous: true)
    nonisolated(unsafe) static let downUpKernel1 = MLXFast.metalKernel(
        name: "track_mixer_down_up_1",
        inputNames: ["normed", "wd", "sd", "bd", "wi", "si", "bi", "wu", "su", "bu", "sigmoid_lut"],
        outputNames: ["lo", "inj", "act", "input", "inject", "inputF"],
        source: downUpSource, header: downUpHeader1, ensureRowContiguous: true)

    /// One launch for the S <= 8 mixer: down and inject, then up and combine.
    /// `lo` is the pre-silu down result and `inj` is the pre-sigmoid inject, both
    /// for the debug taps. `act` is the threadgroup vector published for the up walk.
    static func downUp(
        normed: MLXArray, down: TrackQuantWeight, inject: TrackQuantWeight?, up: TrackQuantWeight,
        hcCount: Int, hidden: Int, hasInject: Bool, emitF32: Bool = false, packedRows: Bool = false
    ) -> (lo: MLXArray, inj: MLXArray, act: MLXArray, input: MLXArray, inject: MLXArray, inputF32: MLXArray) {
        let S = normed.dim(0), KD = normed.dim(1), ND = down.rows
        let HC = inject?.rows ?? hcCount
        precondition(S >= 1 && S <= 8 && ND % 8 == 0 && KD % 512 == 0 && down.bits == 4 && up.bits == 4)
        precondition(hidden % 2 == 0 && up.rows == hcCount * hidden && HC == hcCount)
        precondition(ND % 32 == 0 && ND < 512 + 256)
        precondition(!packedRows || (S == 1 && hcCount == 4))
        precondition(down.biases != nil && up.biases != nil && (inject == nil || inject!.biases != nil))
        let splitEligible = S == 1 && down.groupSize == 32 && (inject == nil || inject!.groupSize == 32)
            && TrackFastMixerSplitK.split > 0
        if splitEligible {
            precondition(TrackFastMixerSplitK.split >= 2, "fused mixer split-K needs two up simdgroups")
        }
        let nsg = splitEligible ? TrackFastMixerSplitK.split : 2
        let injW = inject ?? down
        let sigmoidTable = normed.dtype == .bfloat16 ? TrackBF16Functions.sigmoid : normed
        let outs = (S == 1 ? downUpKernel1 : downUpKernel)(
            [
                normed, down.weight, down.scales, down.biases!, injW.weight, injW.scales, injW.biases!,
                up.weight, up.scales, up.biases!, sigmoidTable,
            ],
            template: [
                ("T", normed.dtype), ("GS", down.groupSize), ("GSU", up.groupSize), ("BITS", down.bits),
                ("KD", KD), ("ND", ND), ("H", hidden), ("HC", HC), ("VPT", S),
                ("HAS_INJECT", hasInject), ("EMIT_F32", emitF32), ("PACKED_ROWS", packedRows),
                ("SPLIT", splitEligible ? TrackFastMixerSplitK.split : 0), ("NSG", nsg),
            ],
            grid: (32, nsg, 1), threadGroup: (32, nsg, 1),
            outputShapes: [[S, ND], [S, HC], [S, ND], [S, hidden], [S, HC], [emitF32 ? S : 1, emitF32 ? hidden : 1]],
            outputDTypes: [normed.dtype, normed.dtype, normed.dtype, normed.dtype, normed.dtype, .float32])
        return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5])
    }
}
