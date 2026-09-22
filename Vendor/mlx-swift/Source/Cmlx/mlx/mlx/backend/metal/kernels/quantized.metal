// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits,                    \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      D,                                                            \
      batched)

#define instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched)               \
  instantiate_kernel(                                                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched,   \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      vecs_per_tg,                                                  \
      k_lanes,                                                      \
      batched)

#define instantiate_quantized_split_k(name, type, group_size, bits, split_k)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose)        \
  instantiate_kernel(                                                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      bm,                                                           \
      bn,                                                           \
      bk,                                                           \
      wm,                                                           \
      wn,                                                           \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits) \
  instantiate_quantized_batched(name, type, group_size, bits, 1)      \
  instantiate_quantized_batched(name, type, group_size, bits, 0)

#define instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_batched_wrap(affine_qmv_fast, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmv, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qvm, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_single(type, group_size, bits) \
  instantiate_quantized(affine_quantize, type, group_size, bits) \
  instantiate_quantized(affine_dequantize, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv_fast, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qvm, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, group_size, bits)   \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, true) \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 0) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 0)

#define instantiate_quantized_all_quad(type, group_size, bits)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 1)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 0)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 1)  \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 0)

// vecs_per_tg (input-vector tile) 2..5; affine uses k_lanes=8 (more rows per
// simdgroup) where the fp path uses 16.
#define instantiate_quantized_wide_wrap(name, type, group_size, bits, vecs_per_tg, k_lanes) \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 0)         \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 1)

#define instantiate_quantized_all_wide(type, group_size, bits) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 2, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 3, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 4, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 5, 8)

#define instantiate_quantized_all_splitk(type, group_size, bits)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 8)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 32)  \

#define instantiate_quantized_splitk_qmm(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                           \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                     \
      type,                                                                     \
      group_size,                                                               \
      bits,                                                                     \
      aligned)

#define instantiate_quantized_all_splitk_qmm(type, group_size, bits)                    \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, true)  \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, false)

#define instantiate_quantized_all_rhs(type, group_size, bits) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 16, 32, 32, 1, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nn, type, group_size, bits, 16, 32, 32, 1, 2, false)

#define instantiate_quantized_funcs(type, group_size, bits) \
  instantiate_quantized_all_single(type, group_size, bits)  \
  instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_all_aligned(type, group_size, bits) \
  instantiate_quantized_all_quad(type, group_size, bits)    \
  instantiate_quantized_all_wide(type, group_size, bits)    \
  instantiate_quantized_all_splitk(type, group_size, bits)  \
  instantiate_quantized_all_splitk_qmm(type, group_size, bits) \
  instantiate_quantized_all_rhs(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)       \
  instantiate_quantized_funcs(float, group_size, bits)      \
  instantiate_quantized_funcs(float16_t, group_size, bits)  \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits) \
  instantiate_quantized_types(128, bits)   \
  instantiate_quantized_types(64, bits)    \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_all() \
  instantiate_quantized_groups(2) \
  instantiate_quantized_groups(3) \
  instantiate_quantized_groups(4) \
  instantiate_quantized_groups(5) \
  instantiate_quantized_groups(6) \
  instantiate_quantized_groups(8)

instantiate_quantized_all()

instantiate_kernel(
    "affine_gather_qmm_gemma4_expert_tiles_bfloat16_t_gs_64_b_4_alN_true_bm_32_bn_32_bk_32",
    affine_gather_qmm_gemma4_expert_tiles,
    bfloat16_t,
    64,
    4,
    true,
    32,
    32,
    32)

// Sorted expert-tile descriptor builders. The E=128 instantiation keeps the
// historical Gemma 4 host name; E=256 serves Qwen 3.5/3.6 MoE. The tile
// kernel instantiation above is expert-count agnostic (K/N are runtime
// arguments) and is shared by both routes.
instantiate_kernel(
    "build_gemma4_sorted_expert_tiles_bm32",
    build_sorted_expert_tiles_bm32,
    128)

instantiate_kernel(
    "build_sorted_expert_tiles_bm32_e256",
    build_sorted_expert_tiles_bm32,
    256)

// Narrow-N split-K quantized GEMM tiles.
//
// `QuantizedMatmul::eval_gpu` sends EVERY transposed 4-bit projection whose
// N is small enough to leave fewer than 512 (N/32 x M/32) tiles through
// `qmm_splitk`, whatever the NAX path would have done. In this engine's wide
// (prefill) window that is three call sites -- `block_inject_weight` (N = 4,
// K = 10240), `in_proj_b`/`in_proj_a` (N = 48) and `shared_expert_gate`
// (N = 1) -- 217 launches per 1024-token chunk. At the stock BN = 32 those
// dispatches carry an N tile eight (or thirty-two) times wider than the
// output: `QuantizedBlockLoader::load_safe` zero-fills the missing rows into
// `Ws`, and `BlockMMA` then runs TN = 2 column fragments per simdgroup with
// at most one of them live. The instantiations below give the same kernel a
// BN that fits the projection.
//
// BIT-EXACTNESS. BN changes neither the operands nor the order in which an
// output element accumulates them: for a fixed (m, n), `BlockMMA::mma` issues
// one `simdgroup_multiply_accumulate` per 8-wide k step in increasing k, the
// k partition boundaries come from the HOST's split_k (which the dispatch
// below keeps computing from the stock 32-wide tiling), each partition is
// still rounded to T on its way to the intermediate, and the col-reduce over
// the partitions is untouched. BM and BK are left at 32 -- BK because
// `QuantizedBlockLoader` static_asserts `BCOLS <= group_size`.
//
// BN must be a multiple of kFragSize * WN = 16 (`BlockMMA` derives TN from
// it) AND `QuantizedBlockLoader` must be able to cover BN rows with its 128
// threads: `n_reads = (BCOLS/pack_factor * BN) / tgp_size` is an integer
// division, so at BK = 32 and 4 bits (BCOLS_PACKED = 4) BN must be either
// below 32 -- where the loader's `bi >= BROWS` guard applies -- or a multiple
// of 32. BN = 16 is the only narrower tile both rules allow, and it is the one
// the measurement wants: it serves N <= 16, which is where the stock BN = 32
// wastes the most (N = 4 fills an eighth of the tile).
#define instantiate_quantized_splitk_qmm_bn(type, group_size, bits, aligned, bn) \
  instantiate_kernel(                                                            \
      "affine_qmm_t_splitk_" #type "_gs_" #group_size "_b_" #bits                \
      "_alN_" #aligned "_bn_" #bn,                                               \
      affine_qmm_t_splitk,                                                       \
      type,                                                                      \
      group_size,                                                                \
      bits,                                                                      \
      aligned,                                                                   \
      32,                                                                        \
      32,                                                                        \
      bn)

instantiate_quantized_splitk_qmm_bn(bfloat16_t, 32, 4, true, 16)
instantiate_quantized_splitk_qmm_bn(bfloat16_t, 32, 4, false, 16)

    // clang-format on
