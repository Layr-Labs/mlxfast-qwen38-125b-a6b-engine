// Copyright © 2023-2025 Apple Inc. Licensed under the MIT license;
// see Vendor/mlx-swift/Source/Cmlx/mlx/LICENSE.

import MLX

extension TrackPrefillMixerAct {
    static let upMixKernel = MLXFast.metalKernel(
        name: "track_prefill_mixer_up_fold",
        inputNames: ["w", "scales", "biases", "x", "normed", "inj"],
        outputNames: ["input", "inject"], source: upMixSource,
        header: TrackPrefillIndirect.metalHeader + TrackFastKernels.exactHeader
            + denseHeader + upMixHeader,
        ensureRowContiguous: true)

    private static let upMixSource = #"""
 const uint3 tid = threadgroup_position_in_grid;
 const uint lid = thread_index_in_threadgroup;
 const uint sg = simdgroup_index_in_threadgroup;
 const uint lane = thread_index_in_simdgroup;
 constexpr int H=2560,HC=4,BM=64,BN=64;
 threadgroup T Ws[64 * 72];
 NAXTile<T,2,2> folded;
 folded.clear();
 for (int hc=0;hc<HC;++hc) {
   uint3 source_tid=tid;
   source_tid.x += hc * (H / BN);
   mixer_up_fold_tile<T,32,4,true,64,64,64,2,2>(
      w,scales,biases,x,normed,folded,Ws,K,N,M,source_tid,lid,sg,lane);
 }
 const int row=tid.y*BM+32*(sg/2), col=tid.x*BN+32*(sg%2);
 folded.store_safe(input+row*H+col,H,short2(32,min(32,M-row)));
 if (HAS_INJECT && tid.x==0) {
   for (int t=lid;t<BM*HC;t+=128) {
     const int r=tid.y*BM+t/HC;
     if (r<M) { inject[r*HC+t%HC]=T(2)*mlx_sigmoid(inj[r*HC+t%HC]); }
   }
 }
"""#

    private static let upMixHeader = #"""
template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void mixer_up_fold_tile(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* normed,
    thread NAXTile<T, BM / WM / 16, BN / WN / 16>& folded,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = TrackMixerActBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      auto run = [&](auto kPrefetch) {
        PackedNAXGroup32 packed_w;
        if constexpr (kPrefetch.value) {
          if (K > 0) {
            packed_w.prefetch(loader_w);
          }
        }
        for (int k = 0; k < K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kPrefetch.value) {
            packed_w.store(loader_w.dst);
          } else if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(short2(BK, tgp_bn));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          if constexpr (kPrefetch.value) {
            if (k + BK < K) {
              loader_w.next();
              packed_w.prefetch(loader_w);
            }
          }

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, TN, TK> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(x + kk1, K);
            } else {
              Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
            }

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<transpose_a>{},
                Btile,
                metal::bool_constant<transpose_b>{});

            (void)compiler_barrier;
          }

          x += BK;
          if constexpr (!kPrefetch.value) {
            loader_w.next();
          }
        }

        // Store results to device memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NAXTile<T, TM, TN> normalized;
        normalized.load_safe(normed + (y_row + tm) * N + y_col + tn,
                             N, short2(sgp_sn, sgp_sm));
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < Dtile.kElemsPerTile; ++e) {
          const T rounded = static_cast<T>(Dtile.elems()[e]);
          const T gate = mlx_sigmoid(rounded);
          const T product = gate * normalized.elems()[e];
          folded.elems()[e] = folded.elems()[e] + product;
        }
      };
      if constexpr (
          metal::is_same_v<T, bfloat16_t> && group_size == 32 && bits == 4 &&
          aligned_N && BM == 64 && BN == 64 && BK == 64 && WM == 2 && WN == 2) {
        dispatch_bool(M > 32 && K % BK == 0, run);
      } else {
        run(metal::false_type{});
      }
    });
  });
}
"""#
}
