// TrackVerify4Exact.swift -- the last two families of a two-row verify window
// that still ran a code path other than the serial S = 1 one, made serial.
//
// A declared draft depth of 1 runs the verify window as ONE [1, 2, ...] forward
// (position 0 = the pending token, position 1 = the draft). The engine commits
// the argmax of row 0 and, when the draft is accepted, the argmax of row 1; the
// official free-run gate then requires the committed sequence to be the serial
// tree's, token for token. That holds only if every kernel in the window is
// per-row bit-identical to the S = 1 kernel it replaces. The router, the MoE
// experts, the shared expert, the HC down+inject and the GDN block already are
// (MLXFAST-V2TWOROW / V3COMPACT / V3GDN); the dense quantized GEMVs are since
// MLXFAST-V4DENSE2 (kernels/quantized.h). This file covers what was left:
//
//   * the HC `up_mix` launch, whose S >= 2 body is the 8-lane `qmv_wide` fold:
//     run the S = 1 kernel once per row (exact by identity, one weight pass
//     per row);
//   * (diagnostic arms, OFF by default) attention appended and attended once
//     per position, and the verify argmax taken by the serial sampler's own
//     `argMax`: both measured token-identical to the engine's serialized
//     attention and top-two kernel, so neither ships on.
//
// The PLE / n-gram block needs nothing: its S >= 2 path is the reference's
// three-launch chain, per-row exact once its two projections take the exact
// M == 2 dense kernel (measured: the free-run gate is green with it, and a
// per-position replay of the S = 1 fusion was tried and withdrawn).
//
// The up_mix valves are marker-file KILL SWITCHES (`/tmp/mlx-v2/no-upmix2row`
// falls back to per-row S = 1 launches, `no-upmix2` to the wide kernel), ON by
// default; a measured box has no such file.

import Foundation
import MLX
import MLXFast

enum TrackVerify4Exact {

    /// Kill switch, ON by default: the two-row `up_mix` kernel (one weight walk,
    /// two accumulators, `qmv_reg_rows`'s own walk per row). Off, the per-row
    /// S = 1 launches serve (`upMixRowsEnabled`), which are exact by identity.
    nonisolated(unsafe) static let upMix2RowEnabled: Bool =
        !FileManager.default.fileExists(atPath: "/tmp/mlx-v2/no-upmix2row")

    /// `qmv_reg_rows` (the S = 1 `up_mix` walk: `qmv_impl`'s normal branch over
    /// four given rows) with TWO activation vectors over one weight walk. Row r's
    /// accumulator sees exactly the sequence of `qdot` results the one-vector
    /// helper produces for that row: same block order, same tail branch, same
    /// scale/bias reads, same device-pointer operands, same closing `simd_sum`.
    static let qmvRegRows2 = #"""
        template <typename T, int group_size, int bits, bool SILU, bool EXACT_TAIL = false>
        METAL_FUNC void qmv_reg_rows2(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            const device T* x1,
            const int in_vec_size,
            const thread int (&rows)[4],
            uint simd_lid,
            thread float (&r0)[4],
            thread float (&r1)[4]) {
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
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
          const device uint8_t* wr[4];
          const device T* sr[4];
          const device T* br[4];
          for (int row = 0; row < 4; row++) {
            wr[row] = ws + rows[row] * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            sr[row] = scales + rows[row] * in_vec_size_g + simd_lid / scale_step_per_thread;
            br[row] = biases + rows[row] * in_vec_size_g + simd_lid / scale_step_per_thread;
          }
          x0 += simd_lid * values_per_thread;
          x1 += simd_lid * values_per_thread;
          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            U sum0 = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x0, x0_thread)
                          : load_vector<T, U, values_per_thread, bits>(x0, x0_thread);
            U sum1 = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x1, x1_thread)
                          : load_vector<T, U, values_per_thread, bits>(x1, x1_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = sr[row][0];
              U b = br[row][0];
              r0[row] += qdot<U, values_per_thread, bits>(wr[row], x0_thread, s, b, sum0);
              r1[row] += qdot<U, values_per_thread, bits>(wr[row], x1_thread, s, b, sum1);
            }
            for (int row = 0; row < 4; row++) {
              wr[row] += block_size * bytes_per_pack / pack_factor;
              sr[row] += block_size / group_size;
              br[row] += block_size / group_size;
            }
            x0 += block_size;
            x1 += block_size;
          }
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
              U sum0 = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x0, x0_thread)
                            : load_vector<T, U, values_per_thread, bits>(x0, x0_thread);
              U sum1 = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x1, x1_thread)
                            : load_vector<T, U, values_per_thread, bits>(x1, x1_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
                U s = sr[row][0];
                U b = br[row][0];
                r0[row] += qdot<U, values_per_thread, bits>(wr[row], x0_thread, s, b, sum0);
                r1[row] += qdot<U, values_per_thread, bits>(wr[row], x1_thread, s, b, sum1);
              }
            }
          } else if (remaining > 0) {
            U sum0 = SILU ? load_vector_safe_silu<T, U, values_per_thread, bits>(x0, x0_thread, remaining)
                          : load_vector_safe<T, U, values_per_thread, bits>(x0, x0_thread, remaining);
            U sum1 = SILU ? load_vector_safe_silu<T, U, values_per_thread, bits>(x1, x1_thread, remaining)
                          : load_vector_safe<T, U, values_per_thread, bits>(x1, x1_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = sr[row][0];
              U b = br[row][0];
              r0[row] += qdot_safe<U, values_per_thread, bits>(wr[row], x0_thread, s, b, sum0, remaining);
              r1[row] += qdot_safe<U, values_per_thread, bits>(wr[row], x1_thread, s, b, sum1, remaining);
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            r0[row] = simd_sum(r0[row]);
            r1[row] = simd_sum(r1[row]);
          }
        }
        """#

    /// The S = 1 `up_mix` body with two activation vectors: tile t owns columns
    /// 2t, 2t+1 across the HC streams (8 rows), each simdgroup four of them,
    /// both rows over one weight walk; the sigmoid-weighted combine below runs
    /// the S = 1 loop once per row. grid (32, 2 * H/2, 1), tg (32, 2, 1).
    static let upMix2RowSource = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const int d0 = 2 * tile;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup float res[8][2];
        int rows[4];
        for (int i = 0; i < 4; ++i) { const int s = (int)sg * 4 + i; rows[i] = d0 + (s & 1) + H * (s >> 1); }
        float r0[4];
        float r1[4];
        qmv_reg_rows2<T, GS, BITS, false, (LW % get_pack_factor<BITS, 32>()) == 0>(wu, su, bu, act, act + LW, LW, rows, lid, r0, r1);
        if (lid == 0) { for (int i = 0; i < 4; ++i) { res[(int)sg * 4 + i][0] = r0[i]; res[(int)sg * 4 + i][1] = r1[i]; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = sg * 32 + lid;
        if (t < 2) {
            const int d = d0 + (int)t;
            for (int v = 0; v < 2; ++v) {
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
        if (HAS_INJECT && tile == 0 && t < (uint)(HC * 2)) {
            const T x = inj[t];
            inject[t] = T(2) * mlx_sigmoid(x);
        }
        """

    nonisolated(unsafe) static let upMix2RowKernel = MLXFast.metalKernel(
        name: "track_mixer_up_mix_2row",
        inputNames: ["act", "normed", "wu", "su", "bu", "inj"],
        outputNames: ["input", "inject", "inputF"],
        source: upMix2RowSource, header: TrackFastMixerKernels.header1 + qmvRegRows2,
        ensureRowContiguous: true)

    /// act [2, LW], normed [2, HC*H], inj [2, HC] -> input [2, H], inject [2, HC], inputF32.
    static func upMix2Row(
        act: MLXArray, normed: MLXArray, up: TrackQuantWeight, inj: MLXArray,
        hcCount: Int, hidden: Int, hasInject: Bool, emitF32: Bool
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray) {
        let S = act.dim(0), LW = act.dim(1)
        precondition(S == 2 && hidden % 2 == 0 && up.rows == hcCount * hidden && up.bits == 4)
        precondition(LW % 32 == 0 && LW < 512 + 256)
        let outs = upMix2RowKernel(
            [act, normed, up.weight, up.scales, up.biases!, inj],
            template: [
                ("T", act.dtype), ("GS", up.groupSize), ("BITS", up.bits), ("H", hidden), ("HC", hcCount),
                ("LW", LW), ("HAS_INJECT", hasInject), ("EMIT_F32", emitF32),
            ],
            grid: (32, (hidden / 2) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[2, hidden], [2, hcCount], [emitF32 ? 2 : 1, emitF32 ? hidden : 1]],
            outputDTypes: [act.dtype, act.dtype, .float32])
        return (outs[0], outs[1], outs[2])
    }

    /// Kill switch, ON by default (see `TrackVerify2TwoRow.routerTwoRowEnabled`).
    nonisolated(unsafe) static let upMixRowsEnabled: Bool =
        !FileManager.default.fileExists(atPath: "/tmp/mlx-v2/no-upmix2")

    /// OFF by default (opt-in marker `/tmp/mlx-v2/attnrows`): attention
    /// update+attend once per row of a two-row window (the decode path's own
    /// L = 1 launch sequence). The engine's own serialized verify attention is
    /// the same launch sequence already (free-run gate identical either way);
    /// the switch exists as a diagnostic arm and measured +0.04 in `c`.
    nonisolated(unsafe) static let attnRowsEnabled: Bool =
        FileManager.default.fileExists(atPath: "/tmp/mlx-v2/attnrows")

    /// OFF by default (opt-in marker `/tmp/mlx-v2/argmaxrows`): the verify
    /// argmax as the serial sampler's own `argMax` over a `[1, V]` float32 copy
    /// of the row instead of the top-two kernel (same tie rule: value desc,
    /// lowest id; free-run gate identical either way). Diagnostic arm.
    nonisolated(unsafe) static let argmaxRowsEnabled: Bool =
        FileManager.default.fileExists(atPath: "/tmp/mlx-v2/argmaxrows")

    /// DIAGNOSTIC (never on a measured box): `/tmp/mlx-v2/diag2` makes every
    /// two-row block recompute itself row by row on the S = 1 path and report
    /// the first bytes that differ, to stderr.
    nonisolated(unsafe) static let diagEnabled: Bool =
        FileManager.default.fileExists(atPath: "/tmp/mlx-v2/diag2")
    nonisolated(unsafe) static var diagReports = 0
    nonisolated(unsafe) static var diagForwards = 0

    static func diagReport(_ tag: String, _ a: MLXArray, _ b: MLXArray) {
        guard a.size == b.size, a.dtype == b.dtype else {
            FileHandle.standardError.write(Data("[diag2] \(tag): SHAPE \(a.shape)/\(a.dtype) vs \(b.shape)/\(b.dtype)\n".utf8))
            return
        }
        let x = a.reshaped(-1).view(dtype: .uint8)
        let y = b.reshaped(-1).view(dtype: .uint8)
        let m = (x .!= y).asType(.int32).sum().item(Int.self)
        if m != 0 && diagReports < 400 {
            diagReports += 1
            FileHandle.standardError.write(Data("[diag2] fwd \(diagForwards) \(tag): \(m) bytes of \(a.size * a.dtype.size)\n".utf8))
        }
    }

    /// `TrackFastMixerKernels.upMix` at S = 1, once per row of a small window.
    /// Every launch is the one-token instantiation (`track_mixer_up_mix_1`), so
    /// row r's output is the serial step's by identity.
    static func upMixRows(
        act: MLXArray, normed: MLXArray, up: TrackQuantWeight, inj: MLXArray,
        hcCount: Int, hidden: Int, hasInject: Bool, emitF32: Bool
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray) {
        let S = act.dim(0)
        precondition(S >= 2 && normed.dim(0) == S && inj.dim(0) == S)
        var inputs: [MLXArray] = []
        var injects: [MLXArray] = []
        var f32s: [MLXArray] = []
        for s in 0 ..< S {
            let u = TrackFastMixerKernels.upMix(
                act: act[s ..< (s + 1)], normed: normed[s ..< (s + 1)], up: up,
                inj: inj[s ..< (s + 1)], hcCount: hcCount, hidden: hidden,
                hasInject: hasInject, emitF32: emitF32)
            inputs.append(u.input)
            injects.append(u.inject)
            f32s.append(u.inputF32)
        }
        return (
            concatenated(inputs, axis: 0), concatenated(injects, axis: 0),
            emitF32 ? concatenated(f32s, axis: 0) : f32s[0]
        )
    }
}
