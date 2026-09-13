// MLXFAST-V3COMPACT (valve): routed-expert SLOT COMPACTION for a two-row
// (depth-1 verify) window, plus the exact two-row shared expert.
//
// At S = 2 the two rows each pick their own top-K experts. The engine launches
// all S*K slots independently, so every expert that BOTH rows selected has its
// weights streamed from DRAM twice. The two routed launches are byte-bound
// (measured 472 GB/s on `gate|up`), so those duplicate bytes are pure loss:
// on a prose tape the two rows' top-10 sets overlap by 3.918 experts on average
// (147 312 adjacent-position pairs), i.e. the union is 16.08 of 20 slots.
//
// This file loads each distinct expert ONCE and runs two independent
// accumulators over the shared weight walk -- the same `qmv_fast_reg2` /
// `qmv_reg2` construction the VERIFY2 round proved bit-exact: row r's block
// walk, scale/bias reads, `qdot` operands (always device loads), accumulation
// order and closing `simd_sum` are exactly the serial S = 1 kernel's. No float
// atomics: each (row, slot) result is written to its own destination, so every
// row keeps its own slot order and its own combine association, and the
// `col_reduce_small` fold over the K experts runs in the same k order as the
// serial path.
//
// Which slot is a duplicate of which is decided IN-KERNEL from the `idx` table
// (K = 10 cached uint loads and one `simd_min` per threadgroup, against a
// ~2 MB weight walk), so no extra dispatch and no host round trip is added.
//
// The shared expert is handled here too: at VPT >= 2 the tip's kernels take
// `qmv_wide_reg_full`, which is NOT the serial S = 1 arithmetic. The two-row
// form below is, and it costs nothing (one expert, one walk).

import Foundation
import MLX
import MLXFast

enum TrackVerify3Compact {

    /// `qmv_reg`'s normal branch (`qmv_impl`) with TWO activation vectors over
    /// ONE weight walk. Row r sees exactly the sequence the one-vector helper
    /// produces for that row, including the MLXFAST-FULLTAIL tail block.
    static let qmvReg2Helpers = #"""

        template <typename T, int group_size, int bits, bool EXACT_TAIL = false, int NR = 4>
        METAL_FUNC void qmv_reg2(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x0,
            const device T* x1,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&r0)[NR],
            thread float (&r1)[NR]) {
          constexpr int results_per_simdgroup = NR;
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
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x0 += simd_lid * values_per_thread;
          x1 += simd_lid * values_per_thread;
          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
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
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
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
            }
          } else if (remaining > 0) {
            U sum0 = load_vector_safe<T, U, values_per_thread, bits>(x0, x0_thread, remaining);
            U sum1 = load_vector_safe<T, U, values_per_thread, bits>(x1, x1_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              r0[row] += qdot_safe<U, values_per_thread, bits>(wl, x0_thread, s, b, sum0, remaining);
              r1[row] += qdot_safe<U, values_per_thread, bits>(wl, x1_thread, s, b, sum1, remaining);
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            r0[row] = simd_sum(r0[row]);
            r1[row] = simd_sum(r1[row]);
          }
        }

        // Slot pairing for a two-row window, decided from `idx` alone.
        //   returns 0xffffffff : this slot's expert is not shared with the other row
        //   returns < 2K       : the partner slot (only ever returned to the row-0 slot)
        //   sets `skip`        : this is the row-1 copy of a shared expert; do nothing
        // Every lane of the simdgroup executes the `simd_min`, so the answer is
        // uniform across the simdgroup and the branch never splits it.
        METAL_FUNC uint track_v3_partner(
            const device uint32_t* idx, int K, int z, int t, uint simd_lid, thread bool& skip) {
          const uint e = idx[z];
          uint cand = 0xffffffffu;
          if (t == 0) {
            if (simd_lid < (uint)K) {
              if (idx[K + (int)simd_lid] == e) { cand = (uint)K + simd_lid; }
            }
            skip = false;
            return simd_min(cand);
          }
          if (simd_lid < (uint)K) {
            if (idx[(int)simd_lid] == e) { cand = simd_lid; }
          }
          skip = simd_min(cand) != 0xffffffffu;
          return 0xffffffffu;
        }
        """#

    static let header =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader
        + TrackFastMoEKernels.regHelpers + TrackVerify2TwoRow.twoRowHelpers + qmvReg2Helpers

    // MARK: gate|up + SwiGLU, slot-compacted.
    //
    // Grid and output layout are the tip's: `act[z]` for routed slot z, then the
    // S shared-expert rows. Only WHICH threadgroup computes a slot changes: the
    // row-0 copy of a shared expert computes both rows over one walk and stores
    // into both slots; the row-1 copy retires immediately.

    static let gateUpCompactSource = """
        const uint z = threadgroup_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint kw = (uint)KD / 8;
        const uint kg = (uint)KD / GS;
        const int out_row = (int)threadgroup_position_in_grid.y * (2 * GRPS)
            + (int)simdgroup_index_in_threadgroup * GRPS;
        if (z == (uint)BR) {
            // Shared expert, both rows, one walk -- and, unlike the tip's
            // `qmv_wide_reg_full` at VPT >= 2, each row is the serial one.
            const device T* x0 = x;
            const device T* x1 = x + (size_t)KD;
            float g0[GRPS], g1[GRPS], u0[GRPS], u1[GRPS];
            qmv_fast_reg2<T, GS, BITS, GRPS>(wsh, ssh, bsh, x0, x1, KD, out_row, lane, g0, g1);
            qmv_fast_reg2<T, GS, BITS, GRPS>(
                wsh + (size_t)N * kw, ssh + (size_t)N * kg, bsh + (size_t)N * kg,
                x0, x1, KD, out_row, lane, u0, u1);
            if (lane == 0) {
                for (int i = 0; i < GRPS; ++i) {
                    act[(size_t)BR * (size_t)N + (size_t)(out_row + i)] =
                        mlx_silu(static_cast<T>(g0[i])) * static_cast<T>(u0[i]);
                    act[(size_t)(BR + 1) * (size_t)N + (size_t)(out_row + i)] =
                        mlx_silu(static_cast<T>(g1[i])) * static_cast<T>(u1[i]);
                }
            }
            return;
        }
        const uint e = idx[z];
        const uint r = xrow[z];
        bool skip = false;
        const uint partner = track_v3_partner(idx, K, (int)z, (int)r, lane, skip);
        if (skip) { return; }
        const size_t eoff = (size_t)e * (size_t)N;
        const device T* xb = x + (size_t)r * (size_t)KD;
        if (partner == 0xffffffffu) {
            float g[GRPS], u[GRPS];
            qmv_fast_reg<T, GS, BITS, GRPS>(
                wg + eoff * kw, sg + eoff * kg, bg + eoff * kg, xb, KD, out_row, lane, g);
            qmv_fast_reg<T, GS, BITS, GRPS>(
                wu + eoff * kw, su + eoff * kg, bu + eoff * kg, xb, KD, out_row, lane, u);
            if (lane == 0) {
                for (int i = 0; i < GRPS; ++i) {
                    act[(size_t)z * (size_t)N + (size_t)(out_row + i)] =
                        mlx_silu(static_cast<T>(g[i])) * static_cast<T>(u[i]);
                }
            }
            return;
        }
        const device T* xp = x + (size_t)xrow[partner] * (size_t)KD;
        float g0[GRPS], g1[GRPS], u0[GRPS], u1[GRPS];
        qmv_fast_reg2<T, GS, BITS, GRPS>(
            wg + eoff * kw, sg + eoff * kg, bg + eoff * kg, xb, xp, KD, out_row, lane, g0, g1);
        qmv_fast_reg2<T, GS, BITS, GRPS>(
            wu + eoff * kw, su + eoff * kg, bu + eoff * kg, xb, xp, KD, out_row, lane, u0, u1);
        if (lane == 0) {
            for (int i = 0; i < GRPS; ++i) {
                act[(size_t)z * (size_t)N + (size_t)(out_row + i)] =
                    mlx_silu(static_cast<T>(g0[i])) * static_cast<T>(u0[i]);
                act[(size_t)partner * (size_t)N + (size_t)(out_row + i)] =
                    mlx_silu(static_cast<T>(g1[i])) * static_cast<T>(u1[i]);
            }
        }
        """

    /// Read a small positive integer out of a marker file (the engine runs as a
    /// separate worker process that does not inherit the launching shell's
    /// environment, so a sweep knob has to be a file).
    static func markerInt(_ name: String, default d: Int) -> Int {
        guard let t = try? String(contentsOfFile: "/tmp/mlx-v2/" + name, encoding: .utf8),
            let v = Int(t.trimmingCharacters(in: .whitespacesAndNewlines)), v > 0
        else { return d }
        return v
    }

    /// Output rows per simdgroup in the compacted `gate|up` kernel. Pure
    /// decomposition: each row's walk, accumulation and `simd_sum` are unchanged
    /// for any value, only which simdgroup owns which rows moves. Two activation
    /// vectors already cost 2 x `values_per_thread` registers, so a smaller value
    /// trades accumulator registers for threadgroups.
    nonisolated(unsafe) static let gateUpRowsPerSimdgroup = markerInt("grps", default: 1)

    nonisolated(unsafe) static let gateUpCompactKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up_compact_2row",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
        outputNames: ["act"],
        source: gateUpCompactSource, header: header, ensureRowContiguous: true)

    /// Drop-in for `TrackFastMoEKernels.gateUpAct` at S = 2 on the decode shapes.
    static func gateUpAct2(
        wg: MLXArray, sg: MLXArray, bg: MLXArray, wu: MLXArray, su: MLXArray, bu: MLXArray,
        shared: TrackQuantWeight, x: MLXArray, idx: MLXArray, xrow: MLXArray,
        groupSize: Int, bits: Int, topK: Int
    ) -> MLXArray {
        let BR = idx.dim(0), S = x.dim(0), KD = x.dim(1), N = wg.dim(1)
        precondition(S == 2 && BR == 2 * topK && N % 8 == 0 && KD % 512 == 0 && bits == 4)
        return gateUpCompactKernel(
            [wg, sg, bg, wu, su, bu, shared.weight, shared.scales, shared.biases!, x, idx, xrow],
            template: [
                ("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("N", N), ("KD", KD),
                ("BR", BR), ("K", topK), ("GRPS", gateUpRowsPerSimdgroup),
            ],
            grid: (32, (N / (2 * gateUpRowsPerSimdgroup)) * 2, BR + 1), threadGroup: (32, 2, 1),
            outputShapes: [[BR + S, N]], outputDTypes: [x.dtype])[0]
    }

    // MARK: down + combine, slot-compacted.
    //
    // The tip gives each token its own threadgroup (grid.z = S), so two rows can
    // never share a `down` walk. Here ONE threadgroup owns the RPS output rows
    // for BOTH tokens and walks the compacted slot list; every product still
    // lands in `prod[t][k]` under its own (token, slot) index and the fold runs
    // in the same k order, so each row's output is the serial row's output.

    static let downCompactSource = """
        const int d0 = (int)threadgroup_position_in_grid.y * RPS;
        const uint sgi = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup float prod[2][K][RPS];
        threadgroup float shvT[2][RPS];
        const uint kw = (uint)F / 8;
        const uint kg = (uint)F / GS;
        for (int zz = 0; zz < (2 * K) / KSG; ++zz) {
            const int z = (int)sgi + zz * KSG;
            const int t = z / K;
            const int k = z - t * K;
            bool skip = false;
            const uint partner = track_v3_partner(idx, K, z, t, lid, skip);
            if (skip) { continue; }
            const uint e = idx[z];
            const size_t eoff = (size_t)e * (size_t)H;
            const device T* xb = act + (size_t)z * (size_t)F;
            const float wk = w[z];
            if (partner == 0xffffffffu) {
                float res[RPS];
                qmv_reg<T, GS, BITS, EXACT_TAIL, RPS>(
                    wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, lid, res);
                if (lid == 0) {
                    for (int i = 0; i < RPS; ++i) {
                        prod[t][k][i] = static_cast<float>(static_cast<T>(res[i])) * wk;
                    }
                }
                continue;
            }
            const device T* xp = act + (size_t)partner * (size_t)F;
            const int kp = (int)partner - K;
            const float wp = w[partner];
            float r0[RPS], r1[RPS];
            qmv_reg2<T, GS, BITS, EXACT_TAIL, RPS>(
                wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, xp, F, d0, lid, r0, r1);
            if (lid == 0) {
                for (int i = 0; i < RPS; ++i) {
                    prod[t][k][i] = static_cast<float>(static_cast<T>(r0[i])) * wk;
                    prod[1][kp][i] = static_cast<float>(static_cast<T>(r1[i])) * wp;
                }
            }
        }
        // Shared expert `down`, both rows, one walk -- serial arithmetic, where
        // the tip's VPT >= 2 path takes `qmv_wide_reg_full`.
        if (sgi == 0) {
            const device T* s0 = act + (size_t)(BR + 0) * (size_t)F;
            const device T* s1 = act + (size_t)(BR + 1) * (size_t)F;
            float q0[RPS], q1[RPS];
            qmv_reg2<T, GS, BITS, EXACT_TAIL, RPS>(wsd, ssd, bsd, s0, s1, F, d0, lid, q0, q1);
            if (lid == 0) {
                for (int i = 0; i < RPS; ++i) {
                    shvT[0][i] = static_cast<float>(static_cast<T>(q0[i]));
                    shvT[1][i] = static_cast<float>(static_cast<T>(q1[i]));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi == 0 && lid == 0) {
            for (int t = 0; t < 2; ++t) {
                const T sgate = mlx_sigmoid(gate[t]);
                for (int i = 0; i < RPS; ++i) {
                    float col[K];
                    for (int k = 0; k < K; ++k) { col[k] = prod[t][k][i]; }
                    const T r = static_cast<T>(mlx_colsum_small_f32<K>(col));
                    const T sh = sgate * static_cast<T>(shvT[t][i]);
                    out[(size_t)t * (size_t)H + (size_t)(d0 + i)] = r + sh;
                }
            }
        }
        """

    nonisolated(unsafe) static let downCompactKernel = MLXFast.metalKernel(
        name: "track_moe_down_combine_compact_2row",
        inputNames: ["wd", "sd", "bd", "wsd", "ssd", "bsd", "act", "idx", "w", "gate"],
        outputNames: ["out"],
        source: downCompactSource, header: header, ensureRowContiguous: true)

    /// Output rows per down+combine threadgroup in the compacted two-row kernel.
    /// Pure decomposition: each row's walk, accumulation and fold are unchanged.
    nonisolated(unsafe) static let downRowsPerThreadgroup = markerInt("drps", default: 2)
    nonisolated(unsafe) static let downSimdgroups = markerInt("dksg", default: 5)

    /// Drop-in for `TrackFastMoEKernels.downCombine` at S = 2 on the decode shapes.
    static func downCombine2(
        wd: MLXArray, sd: MLXArray, bd: MLXArray, sharedDown: TrackQuantWeight, act: MLXArray,
        idx: MLXArray, w: MLXArray, gate: MLXArray, topK: Int, groupSize: Int, bits: Int
    ) -> MLXArray {
        let BR = idx.dim(0), F = act.dim(1), H = wd.dim(1)
        precondition(BR == 2 * topK && H % downRowsPerThreadgroup == 0 && bits == 4)
        precondition(act.dim(0) == BR + 2 && gate.dim(0) == 2 && sharedDown.rows == H)
        let ksg = (2 * topK) % downSimdgroups == 0 ? downSimdgroups : 1
        let rps = downRowsPerThreadgroup
        let packFactor = 32 / bits
        return downCompactKernel(
            [wd, sd, bd, sharedDown.weight, sharedDown.scales, sharedDown.biases!, act, idx, w, gate],
            template: [
                ("T", act.dtype), ("GS", groupSize), ("BITS", bits), ("H", H), ("F", F),
                ("K", topK), ("BR", BR), ("KSG", ksg), ("RPS", rps),
                ("EXACT_TAIL", F % packFactor == 0),
            ],
            grid: (32, (H / rps) * ksg, 1), threadGroup: (32, ksg, 1),
            outputShapes: [[2, H]], outputDTypes: [act.dtype])[0]
    }

    /// Kill switch. ON by default; a marker file turns it OFF, so one build
    /// serves both arms of a local A/B. A `var` so the bit-comparison harness can
    /// exercise the `GATE2ROW` branch of `routeSource` in-process; the engine
    /// never writes it, and a measured box has no `/tmp/mlx-v2`.
    nonisolated(unsafe) static var enabled: Bool =
        !FileManager.default.fileExists(atPath: "/tmp/mlx-v2/no-moecompact")
}
