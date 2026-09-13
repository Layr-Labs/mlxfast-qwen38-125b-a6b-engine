// Copyright © 2023-2025 Apple Inc. Licensed under the MIT license;
// see Vendor/mlx-swift/Source/Cmlx/mlx/LICENSE.

import MLX

extension TrackPrefillIndirect {
    private static let scatterDownKernel = MLXFast.metalKernel(
        name: "track_prefill_down_output_scatter",
        inputNames: ["x", "w", "scales", "biases", "indices", "order"],
        outputNames: ["y"], source: scatterDownSource,
        header: metalHeader + scatterDownHeader, ensureRowContiguous: true)

    static func scatterDown(
        _ m: TrackMoE, activated: MLXArray, sortedIDs: MLXArray, order: MLXArray
    ) -> MLXArray? {
        let rows = order.size
        let d = m.expertDown
        guard m.expertBits == 4, m.expertGroupSize == 32,
            activated.shape == [rows, 1, 640], activated.dtype == .bfloat16,
            rows >= 2048, rows < 512 * 64, rows % 10 == 0,
            sortedIDs.shape == [rows], sortedIDs.dtype == .uint32,
            order.shape == [rows], order.dtype == .uint32,
            d.w.shape == [512, 2560, 80], d.w.dtype == .uint32,
            d.s.shape == [512, 2560, 20], d.s.dtype == .bfloat16,
            d.b.shape == d.s.shape, d.b.dtype == .bfloat16
        else { return nil }
        return scatterDownKernel(
            [activated, d.w, d.s, d.b, sortedIDs, order],
            template: [("T", activated.dtype), ("M", rows), ("N", 2560), ("K", 640)],
            grid: (40 * 32, ((rows + 31) / 32) * 2, 2),
            threadGroup: (32, 2, 2),
            outputShapes: [[rows, 1, 2560]], outputDTypes: [.bfloat16])[0]
    }

    private static let scatterDownSource = #"""
        threadgroup T Ws[64 * 72];
        threadgroup T As[32 * 72];
        scatter_down_nax<T, 32, 4, 32, 64, 64, 2, 2, true>(
            x, w, scales, biases, indices, order, y,
            M, N, K, Ws, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

    private static let scatterDownHeader = #"""
template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
METAL_FUNC void scatter_down_nax(
    const device T* x,
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device uint32_t* indices,
    const device uint32_t* order,
    device T* y,
    const int M,
    const int N,
    const int K,
    threadgroup T* Ws,
    threadgroup T* As,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  static_assert(
      transpose && BM == 32 && BN == 64 && BK == 64 && WM == 2 && WN == 2,
      "P17 requires the original 32x64x64 NAX tile and 2x2 SIMD layout");
  static_assert(
      metal::is_same_v<T, bfloat16_t> && group_size == 32 && bits == 4,
      "P17 requires unchanged bf16 / affine group-32 / 4-bit operands");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  // MLXFAST-ASTAGE: the A tile gets the same padded leading dimension Ws uses.
  constexpr int BKA_padded = BK_padded;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, transpose, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const short tgp_bm = short(min(BM, M - y_row));

  auto wl = (const device uint8_t*)w;
  wl += size_t(y_col) * K_w;
  scales += size_t(y_col) * K_g;
  biases += size_t(y_col) * K_g;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  constexpr short BR = TN;
  constexpr short BC = TK;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  using AccumType = float;

  // This scan is the original sorted-RHS scan, not a new sort or permutation.
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    int tile_begin;
    int tile_end;
    p17_sorted_expert_tile<BM>(
        indices, M, y_row, offset, offset_next, index, tile_begin, tile_end);
    if (tile_begin == tile_end) {
      continue;  // Uniform over the ENTIRE threadgroup; no barrier is skipped by a subset.
    }
    const short tile_m = short(tile_end - tile_begin);
    const short sgp_sm = short(min(int(SM), max(0, int(tile_m) - int(tm))));
    const bool sg_active = sgp_sm > 0;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();
    // MLXFAST-ASTAGE: one threadgroup-uniform base. The A rows this tile needs
    // are staged cooperatively, so no simdgroup walks the device pointer itself.
    const device T* xb = x + size_t(tile_begin) * K;
    const short tgp_thread = short(simd_group_id * SIMD_SIZE + simd_lane_id);
    const short a_row = tgp_thread / 4;            // 0..BM-1
    const short a_col = (tgp_thread % 4) * 16;     // 0,16,32,48
    threadgroup T* a_dst = As + a_row * BKA_padded + a_col;
    const bool a_live = a_row < tile_m;

    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        K,
        Ws,
        simd_group_id,
        simd_lane_id);

    // N and K alignment are runtime-gated at the call site. Row tails only
    // change load/store predicates, never the per-output K reduction.
    // This specialization is threadgroup-uniform. A partial BM tile uses
    // safe loads/stores even for a full first SM, with identical live values.
    dispatch_bool(tile_m == BM, [&](auto kAlignedM) {
      // MLXFAST-APREFETCH: the A block is fetched into registers one K block
      // ahead, so its device latency is hidden behind the current block's MMAs
      // instead of standing between the two barriers. This is the same software
      // pipeline `PackedNAXGroup32` already gives the weight block; A just did
      // not have one. 16 bf16 per thread, statically indexed, so it stays in
      // registers. The values published to `As` are byte for byte what the
      // in-place copy published, only fetched earlier.
      T a_buf[16];
      PackedNAXGroup32 packed_w;
      if (K_it > 0) {
        packed_w.prefetch(loader_w);
        if (a_live) {
          const device T* a0 = xb + size_t(a_row) * K + a_col;
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 16; ++e) { a_buf[e] = a0[e]; }
        }
      }
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        packed_w.store(loader_w.dst);
        // MLXFAST-ASTAGE. Each simdgroup used to walk A itself: 16 rows of 64
        // bytes at a 5,120-byte stride, per simdgroup, per kk1 -- and with
        // WM = WN = 2 the pairs (0,1) and (2,3) issue IDENTICAL reads, so every
        // A row of the tile is fetched twice. Staging it instead costs one
        // cooperative, fully coalesced pass: 128 threads x 16 contiguous
        // elements covers the whole BM x BK block, four threads to a row, one
        // 128-byte line per row. It rides the barrier pair Ws already needs, so
        // it adds no synchronization. Rows past `tile_m` are zeroed, which is
        // what the `load_safe` path they replace produces for the same lanes;
        // an out-of-range row can only ever reach its own Dtile row, and those
        // rows are excluded by `store_slice` either way. Same values, same
        // order, bit-identical output.
        if (a_live) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 16; ++e) {
            a_dst[e] = a_buf[e];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 16; ++e) {
            a_dst[e] = T(0);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // All lanes prefetch; Ws is not reused until the next reader barrier.
        if (k + 1 < K_it) {
          loader_w.next();
          packed_w.prefetch(loader_w);
          if (a_live) {
            const device T* a_next = xb + BK + size_t(a_row) * K + a_col;
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 16; ++e) { a_buf[e] = a_next[e]; }
          }
        }

        STEEL_PRAGMA_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            Atile.template load<T, BKA_padded, 1>(
                As + tm * BKA_padded + kk1);

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        xb += BK;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        const short2 sc = BaseNAXFrag::get_coord();
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
          STEEL_PRAGMA_UNROLL
          for (short r = 0; r < BaseNAXFrag::kElemRows; ++r) {
            const short local_row = i * 16 + sc.y + r * BaseNAXFrag::kElemRowsJump;
            if (local_row < sgp_sm) {
              const size_t dst_row = order[tile_begin + tm + local_row];
              device T* dst = y + dst_row * N + y_col + tn + sc.x;
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < TN; ++j) {
                STEEL_PRAGMA_UNROLL
                for (short c = 0; c < BaseNAXFrag::kElemCols; ++c) {
                  dst[j * 16 + c] = static_cast<T>(Dtile.frag_at(i, j)[r * BaseNAXFrag::kElemCols + c]);
                }
              }
            }
          }
        }
      }
    });
  }
}

"""#
}
