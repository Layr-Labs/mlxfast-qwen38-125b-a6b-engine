// MLXFAST-INJFUSE: One-token mixers fuse injectNorm into down/inject (two launches).
// Standalone norms (PLE) and S=2..8 retain the original three-launch chain:
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
    static let header =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + TrackFastMoEKernels.regHelpers
        + TrackFastKernels.mixerHeadHeaderTail + TrackFastMoEKernels.wideHelpers
    /// One-token instantiations: the wide bodies are replaced by their declarations.
    static let header1 =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + TrackFastMoEKernels.regHelpers
        + TrackFastKernels.mixerHeadHeaderTail + TrackFastMoEKernels.wideDecls

    // MLXFAST-INJFUSE: Verbatim load_vector/load_vector_safe and qmv_fast_reg
    // from TrackFastMoE.swift, and track_inject_qmv from TrackFastKernels2.swift.
    // Only each x parameter's address space changes; qdot/qdot_safe stay shared.
    static let threadgroupHelpers = #"""
        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector(const threadgroup T* x, thread U* x_thread) {
          static_assert(
              bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
                  bits == 8,
              "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

          U sum = 0;

          if (bits == 2) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 4.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 64.0f;
            }
          }

          else if (bits == 3) {
            for (int i = 0; i < values_per_thread; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 8.0f;
              x_thread[i + 2] = x[i + 2] / 64.0f;
              x_thread[i + 3] = x[i + 3] / 2.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 128.0f;
              x_thread[i + 6] = x[i + 6] / 4.0f;
              x_thread[i + 7] = x[i + 7] / 32.0f;
            }
          }

          else if (bits == 4) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 16.0f;
              x_thread[i + 2] = x[i + 2] / 256.0f;
              x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
          }

          else if (bits == 5) {
            for (int i = 0; i < values_per_thread; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 32.0f;
              x_thread[i + 2] = x[i + 2] / 4.0f;
              x_thread[i + 3] = x[i + 3] / 128.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 2.0f;
              x_thread[i + 6] = x[i + 6] / 64.0f;
              x_thread[i + 7] = x[i + 7] / 8.0f;
            }
          }

          else if (bits == 6) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 64.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 4.0f;
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < values_per_thread; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }

          return sum;
        }

        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_safe(const threadgroup T* x, thread U* x_thread, int N) {
          static_assert(
              bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
                  bits == 8,
              "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

          U sum = 0;

          if (bits == 2) {
            for (int i = 0; i < N; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 4.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 64.0f;
            }
          }

          else if (bits == 3) {
            for (int i = 0; i < N; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];

              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 8.0f;
              x_thread[i + 2] = x[i + 2] / 64.0f;
              x_thread[i + 3] = x[i + 3] / 2.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 128.0f;
              x_thread[i + 6] = x[i + 6] / 4.0f;
              x_thread[i + 7] = x[i + 7] / 32.0f;
            }
          }

          else if (bits == 4) {
            for (int i = 0; i < N; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 16.0f;
              x_thread[i + 2] = x[i + 2] / 256.0f;
              x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
          }

          else if (bits == 5) {
            for (int i = 0; i < N; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 32.0f;
              x_thread[i + 2] = x[i + 2] / 4.0f;
              x_thread[i + 3] = x[i + 3] / 128.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 2.0f;
              x_thread[i + 6] = x[i + 6] / 64.0f;
              x_thread[i + 7] = x[i + 7] / 8.0f;
            }
          }

          else if (bits == 6) {
            for (int i = 0; i < N; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 64.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 4.0f;
            }
          }

          else if (bits == 8) {
            for (int i = 0; i < N; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }

          for (int i = N; i < values_per_thread; i++) {
            x_thread[i] = 0;
          }

          return sum;
        }

        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_fast_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const threadgroup T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) { result[row] = 0; }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          for (int k = 0; k < in_vec_size; k += block_size) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        template <typename T, int group_size, int bits, int in_vec_size, int out_vec_size, int UNR>
        METAL_FUNC void track_inject_qmv(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const threadgroup T* x,
            device T* y,
            uint simd_gid,
            uint simd_lid) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(out_vec_size < num_simdgroups * results_per_simdgroup, "small-N branch only");
          static_assert(in_vec_size > block_size, "K walk");

          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          thread U result[results_per_simdgroup] = {0};

          constexpr int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          constexpr int in_vec_size_g = in_vec_size / group_size;
          const int out_row = simd_gid * results_per_simdgroup;
          if (out_row >= out_vec_size) {
            return;
          }
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          y += out_row;

          // for (k = 0; k < in_vec_size - block_size; k += block_size)
          constexpr int NFULL = (in_vec_size - 1) / block_size;
          // simd_gid 0 only reaches here, so out_row == 0 and the row count is
          // compile-time: same rows, same order, but the compiler can batch
          // the loads (the runtime-bounded loop in `qmv_impl` serializes them).
          constexpr int NR = out_vec_size < results_per_simdgroup ? out_vec_size : results_per_simdgroup;
          #pragma clang loop unroll_count(UNR)
          for (int i = 0; i < NFULL; i++) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }
          constexpr int k_end = NFULL * block_size;
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k_end - simd_lid * values_per_thread), 0, values_per_thread);
          if (remaining > 0) {
            U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < NR; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) {
              y[row] = static_cast<T>(result[row]);
            }
          }
        }
        """#

    // MLXFAST-INJFUSE: Reuse one bf16 buffer for stream, then normed. Each
    // existing (32,2,1) threadgroup repeats all four norms: sg0 owns hc0/1,
    // sg1 owns hc2/3. Norm temporaries end before either GEMV walk begins.
    // H=2560, HC=4: 20,480 bytes + 256 bytes of independent reduction scratch.
    // Arithmetic is copied from TrackFastKernels.injectNormWideSource; only
    // the pending-inject flag, storage, ownership and synchronization change.
    static let downInjectNormSource = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        threadgroup T normedTG[KD];
        threadgroup float sums[2][32];
        {
            using InT = T;
            constexpr int N_READS = 4;
            constexpr uint NT = H / N_READS;
            constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
            constexpr uint row = 0;
            constexpr uint W = KD;
            threadgroup float* local_sums = sums[sg];
            for (uint hc = sg * 2; hc < sg * 2 + 2; ++hc) {
                const uint base = row * W + hc * H;
                InT inj_t = InT(0);
                if (NORM_HAS_INJECT) { inj_t = inject[row * HC + hc]; }
                for (uint g = 0; g < simd_groups; ++g) {
                    const uint lid = g * 32 + lane;
                    float acc = 0.0f;
                    if (lid < NT) {
                        for (int i = 0; i < N_READS; ++i) {
                            const uint d = lid * N_READS + i;
                            const uint src = TILE ? (row * H + d) : (base + d);
                            InT r = residual[src];
                            if (NORM_HAS_INJECT) {
                                InT sp = out[row * H + d] * inj_t;
                                r = r + sp;
                            }
                            normedTG[base + d] = r;
                            if (tile == 0) { stream[base + d] = r; }
                            const float xf = static_cast<float>(r);
                            acc += xf * xf;
                        }
                    }
                    acc = simd_sum(acc);
                    if (lane == 0) { local_sums[g] = acc; }
                }
                if (lane >= simd_groups) { local_sums[lane] = 0; }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                const float total = simd_sum(local_sums[lane]);
                const float inv_mean = metal::precise::rsqrt(total / (float)H + as_type<float>((uint)EPS_BITS));
                for (uint g = 0; g < simd_groups; ++g) {
                    const uint lid = g * 32 + lane;
                    if (lid < NT) {
                        for (int i = 0; i < N_READS; ++i) {
                            const uint d = lid * N_READS + i;
                            InT n = static_cast<InT>(static_cast<float>(normedTG[base + d]) * inv_mean);
                            normedTG[base + d] = n * scale[hc * H + d];
                            if (tile == 0) { normed[base + d] = normedTG[base + d]; }
                        }
                    }
                }
                // MLXFAST-INJFUSE: Finish all readers before reusing sums[sg].
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // MLXFAST-INJFUSE: Original VPT=1 walks, with x in threadgroup memory.
        const uint lid = lane;
        constexpr int NT = ND / 8;
        if (tile < NT) {
            float r[4];
            qmv_fast_reg<T, GS, BITS>(wd, sd, bd, normedTG, KD, tile * 8 + (int)sg * 4, lid, r);
            if (lid == 0) {
                for (int i = 0; i < 4; ++i) {
                    const T l = static_cast<T>(r[i]);
                    lo[tile * 8 + (int)sg * 4 + i] = l;
                    act[tile * 8 + (int)sg * 4 + i] = mlx_silu(l);
                }
            }
        } else if (HAS_INJECT) {
            track_inject_qmv<T, GS, BITS, KD, HC, 4>(wi, si, bi, normedTG, inj, sg, lid);
        }
        """

    /// normed [S, KD] -> lo [S, ND] (down), inj [S, HC] (inject).
    /// grid threads (32, 2 * (ND/8 + (HAS_INJECT ? 1 : 0)), 1), tg (32, 2, 1).
    static let downInjectSource = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        constexpr int NT = ND / 8;
        if (tile < NT) {
            if constexpr (VPT == 1) {
                float r[4];
                qmv_fast_reg<T, GS, BITS>(wd, sd, bd, normed, KD, tile * 8 + (int)sg * 4, lid, r);
                if (lid == 0) {
                    for (int i = 0; i < 4; ++i) {
                        const T l = static_cast<T>(r[i]);
                        lo[tile * 8 + (int)sg * 4 + i] = l;
                        act[tile * 8 + (int)sg * 4 + i] = mlx_silu(l);
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
                track_inject_qmv<T, GS, BITS, KD, HC, 4>(wi, si, bi, normed, inj, sg, lid);
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
    // MLXFAST-INJFUSE: Retain the original one-token kernel for standalone norms (PLE).
    nonisolated(unsafe) static let downInjectNormedKernel1 = MLXFast.metalKernel(
        name: "track_mixer_down_inject_normed_1",
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
        let tiles = ND / 8 + (inject != nil ? 1 : 0)
        // MLXFAST-INJFUSE: Already-normalized one-token and S=2..8 paths are unchanged.
        let outs = (S == 1 ? downInjectNormedKernel1 : downInjectKernel)(
            [normed, down.weight, down.scales, down.biases!, inj.weight, inj.scales, inj.biases!],
            template: [
                ("T", normed.dtype), ("GS", down.groupSize), ("BITS", down.bits), ("KD", KD), ("ND", ND),
                ("HC", HC), ("VPT", S), ("HAS_INJECT", inject != nil),
            ],
            grid: (32, tiles * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[S, ND], [S, ND], [S, HC]], outputDTypes: [normed.dtype, normed.dtype, normed.dtype])
        return (outs[0], outs[1], outs[2])
    }

    // MLXFAST-INJFUSE: Same grid and threadgroup as downInject; no norm launch.
    nonisolated(unsafe) static let downInjectKernel1 = MLXFast.metalKernel(
        name: "track_mixer_down_inject_1",
        inputNames: ["residual", "out", "inject", "scale", "wd", "sd", "bd", "wi", "si", "bi"],
        outputNames: ["lo", "act", "inj", "stream", "normed"],
        source: downInjectNormSource, header: header1 + threadgroupHelpers, ensureRowContiguous: true)

    // MLXFAST-INJFUSE: NORM_HAS_INJECT describes the pending residual update;
    // HAS_INJECT independently describes this mixer's outgoing inject projection.
    static func downInject(
        residual: MLXArray, out: MLXArray?, pendingInject: MLXArray?, scale: MLXArray,
        down: TrackQuantWeight, inject: TrackQuantWeight?, hcCount: Int, hidden: Int,
        eps: Float, tile: Bool
    ) -> (lo: MLXArray, act: MLXArray, inj: MLXArray, stream: MLXArray, normed: MLXArray) {
        let KD = hcCount * hidden, ND = down.rows
        precondition(residual.dim(0) == 1 && residual.dim(1) == 1 && residual.dtype == .bfloat16)
        precondition(hcCount == 4 && hidden == 2560 && ND % 8 == 0 && down.bits == 4)
        precondition(residual.dim(2) == (tile ? hidden : KD) && scale.size == KD)
        precondition((out == nil) == (pendingInject == nil))
        if let inject {
            precondition(inject.rows == hcCount && inject.bits == down.bits
                && inject.groupSize == down.groupSize)
        }
        let inj = inject ?? down
        let tiles = ND / 8 + (inject != nil ? 1 : 0)
        let outs = downInjectKernel1(
            [residual, out ?? residual, pendingInject ?? residual, scale,
             down.weight, down.scales, down.biases!, inj.weight, inj.scales, inj.biases!],
            template: [
                ("T", residual.dtype), ("GS", down.groupSize), ("BITS", down.bits), ("KD", KD), ("ND", ND),
                ("HC", hcCount), ("H", hidden), ("HAS_INJECT", inject != nil),
                ("NORM_HAS_INJECT", out != nil), ("TILE", tile), ("EPS_BITS", Int(eps.bitPattern)),
            ],
            grid: (32, tiles * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, ND], [1, ND], [1, hcCount], [1, 1, KD], [1, 1, KD]],
            outputDTypes: Array(repeating: residual.dtype, count: 5))
        return (outs[0], outs[1], outs[2], outs[3], outs[4])
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
            qmv_reg_rows<T, GS, BITS, false>(wu, su, bu, act, LW, rows, lid, r);
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
