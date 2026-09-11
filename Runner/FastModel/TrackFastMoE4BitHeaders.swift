// MLXFAST-HDRTRIM
// Literal 4-bit variants of TrackFastMoEKernels' full-width helper headers.
// The retained branch bodies are copied character-for-character. Only dead
// width branches, their selectors, and template rejection guards differ.
// Keep these copies in step with helpersCore / regHelpers / wideHelpers.
// The full-width originals remain available, but every current custom-kernel
// consumer has a Swift bits == 4 precondition and uses these variants.
// These are static literals: trimming does no work on the decode path.

extension TrackFastMoEKernels {
    static let helpersCore4 = #"""
#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");
  return wsize / bits;
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");
  return (wsize / 8);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");

  U sum = 0;

  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");

  U sum = 0;

  if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");

  U accum = 0;

  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");

  U accum = 0;

  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  return scale * accum + sum * bias;
}

"""#

    // MLXFAST-HDRTRIM: get_pack_factor / get_bytes_per_pack in helpersCore4
    // reject non-4-bit instantiations of the register helpers as well.
    static let regHelpers4 = #"""

        // qmv_fast_impl with `out_row` given and the 4 row results returned
        // (all lanes hold them after simd_sum). x points at the vector.
        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_fast_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int packs_per_thread = 2;
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

        // qmv_impl's normal branch (out_vec_size >= 8, full tile) likewise.
        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
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
          const int used_out_row = out_row;
          ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
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
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
          if (remaining > 0) {
            U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        // load_vector / load_vector_safe (4-bit) with `silu` applied to each
        // activation on load: the same bf16 silu the separate launch stored.
        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_silu(const device T* x, thread U* x_thread) {
          static_assert(bits == 4, "silu-on-load: 4-bit only");
          U sum = 0;
          for (int i = 0; i < values_per_thread; i += 4) {
            const T a = mlx_silu(x[i]);
            const T b = mlx_silu(x[i + 1]);
            const T c = mlx_silu(x[i + 2]);
            const T d = mlx_silu(x[i + 3]);
            sum += a + b + c + d;
            x_thread[i] = a;
            x_thread[i + 1] = b / 16.0f;
            x_thread[i + 2] = c / 256.0f;
            x_thread[i + 3] = d / 4096.0f;
          }
          return sum;
        }
        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_safe_silu(const device T* x, thread U* x_thread, int N) {
          static_assert(bits == 4, "silu-on-load: 4-bit only");
          U sum = 0;
          for (int i = 0; i < N; i += 4) {
            const T a = mlx_silu(x[i]);
            const T b = mlx_silu(x[i + 1]);
            const T c = mlx_silu(x[i + 2]);
            const T d = mlx_silu(x[i + 3]);
            sum += a + b + c + d;
            x_thread[i] = a;
            x_thread[i + 1] = b / 16.0f;
            x_thread[i + 2] = c / 256.0f;
            x_thread[i + 3] = d / 4096.0f;
          }
          for (int i = N; i < values_per_thread; i++) {
            x_thread[i] = 0;
          }
          return sum;
        }

        // qmv_impl's normal branch over FOUR GIVEN rows (each row's walk is
        // independent of its neighbours), optional silu on the activations.
        template <typename T, int group_size, int bits, bool SILU>
        METAL_FUNC void qmv_reg_rows(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const thread int (&rows)[4],
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
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
          const device uint8_t* wr[4];
          const device T* sr[4];
          const device T* br[4];
          for (int row = 0; row < 4; row++) {
            wr[row] = ws + rows[row] * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            sr[row] = scales + rows[row] * in_vec_size_g + simd_lid / scale_step_per_thread;
            br[row] = biases + rows[row] * in_vec_size_g + simd_lid / scale_step_per_thread;
          }
          x += simd_lid * values_per_thread;
          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            U sum = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x, x_thread)
                         : load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = sr[row][0];
              U b = br[row][0];
              result[row] += qdot<U, values_per_thread, bits>(wr[row], x_thread, s, b, sum);
            }
            for (int row = 0; row < 4; row++) {
              wr[row] += block_size * bytes_per_pack / pack_factor;
              sr[row] += block_size / group_size;
              br[row] += block_size / group_size;
            }
            x += block_size;
          }
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
          if (remaining > 0) {
            U sum = SILU ? load_vector_safe_silu<T, U, values_per_thread, bits>(x, x_thread, remaining)
                         : load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = sr[row][0];
              U b = br[row][0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wr[row], x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }
        """#

    static let wideHelpers4 = #"""
        // MLX `qmv_wide_impl` (the M >= 2 quantized GEMV on this GPU generation),
        // verbatim for a FULL 8-row tile (k_folds == 1): same lane -> group
        // assignment (k_lane, stride k_lanes), same per-group decode and
        // element-order accumulation, same shuffle ladder. The row is given by
        // the caller; the vecs_per_tg vectors are x's first rows. The totals
        // are left in `result` on the k_lane == 0 lanes.
        template <typename T> METAL_FUNC vec<T, 4> track_silu4(vec<T, 4> x) {
            return vec<T, 4>(mlx_silu(x.x), mlx_silu(x.y), mlx_silu(x.z), mlx_silu(x.w));
        }
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, bool SILU>
        METAL_FUNC void qmv_wide_reg_full(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int M,
            const int row,
            uint simd_lid,
            thread float (&result)[vecs_per_tg]) {
          constexpr int sub = 8; // values per sub-chunk (== bits bytes, byte-aligned)
          typedef float U;
          const short k_lane = simd_lid % k_lanes;
          constexpr int k_folds = 1;
          constexpr int fold = 0;
          constexpr int vec0 = 0;
        
          const int in_vec_size_w = in_vec_size * bits / 8; // bytes per weight row
          const int in_vec_size_g = in_vec_size / group_size;
          const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
          const device T* srow = scales + row * in_vec_size_g;
          const device T* brow = biases + row * in_vec_size_g;
        
          const device T* xv[vecs_per_tg];
          for (int v = 0; v < vecs_per_tg; v++) {
            xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
          }
        
          for (int v = 0; v < vecs_per_tg; v++) { result[v] = 0; }
        
          // Each lane reduces a strided subset of the row's groups: decode the group in
          // 8-value sub-chunks and reuse each chunk across the streamed vectors.
          const int g_stride = k_lanes * k_folds;
          const int g_first = fold < k_folds ? k_lane + fold * k_lanes : in_vec_size_g;
          // Group loop. The 4-bit path is unrolled by g_unroll: the scale, the bias
          // and the packed weights of every group a trip covers are loaded before any
          // of them is decoded, so g_unroll independent memory requests per thread are
          // in flight at once. At the decode output widths only 40-80 threadgroups are
          // resident on the whole GPU, so there is no other thread to hide a load
          // behind and the stock loop paid one memory round trip per group. Every
          // group is decoded by the same expression as before, every vector sums its
          // terms in ascending element order, and the group partials still reach
          // result[v] in ascending group order, so the arithmetic is bit identical.
          static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");
          if constexpr (bits == 4) {
            constexpr int g_unroll = 3;
            constexpr int packs_per_group = group_size / sub;
            int g = g_first;
            for (; g + (g_unroll - 1) * g_stride < in_vec_size_g;
                 g += g_unroll * g_stride) {
              float su[g_unroll];
              float bu[g_unroll];
              uint32_t wpack[g_unroll][packs_per_group];
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                su[u] = static_cast<float>(srow[gu]);
                bu[u] = static_cast<float>(brow[gu]);
                const device uint32_t* wg =
                    (const device uint32_t*)(wrow + gu * (group_size * bits / 8));
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  wpack[u][sc] = wg[sc];
                }
              }
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                const float s = su[u];
                const float b = bu[u];
                const float s_hi = s / 16.0f;
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  const int k0 = gu * group_size + sc * sub;
                  const uint32_t p = wpack[u][sc];
                  U w_dq[sub];
        #pragma unroll
                  for (int i = 0; i < sub / 2; i++) {
                    const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                    w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                    w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                  }
                  // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                  // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                  // of eight, issued before any product. The element-major order below is
                  // unchanged, so every vector still sums its terms in ascending element
                  // order -- bit identical.
                  vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                    for (int c = 0; c < sub / 4; c++) {
                      xq[v][c] = SILU ? track_silu4<T>(xc4[c]) : xc4[c];
                    }
                  }
                  U accv[vecs_per_tg] = {0};
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                    }
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    result[v] += accv[v];
                  }
                }
              }
            }
            // Groups left over when the row's group count is not a multiple of
            // g_unroll, in the same ascending order.
            for (; g < in_vec_size_g; g += g_stride) {
              const float s = static_cast<float>(srow[g]);
              const float b = static_cast<float>(brow[g]);
              const float s_hi = s / 16.0f;
              const device uint32_t* wg =
                  (const device uint32_t*)(wrow + g * (group_size * bits / 8));
              uint32_t wpack[group_size / sub];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                wpack[sc] = wg[sc];
              }
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const uint32_t p = wpack[sc];
                U w_dq[sub];
        #pragma unroll
                for (int i = 0; i < sub / 2; i++) {
                  const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                  w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                  w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                }
                // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                // of eight, issued before any product. The element-major order below is
                // unchanged, so every vector still sums its terms in ascending element
                // order -- bit identical.
                vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
                    xq[v][c] = SILU ? track_silu4<T>(xc4[c]) : xc4[c];
                  }
                }
                U accv[vecs_per_tg] = {0};
        #pragma unroll
                for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                  }
                }
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  result[v] += accv[v];
                }
              }
            }
          }
          // Reduce each vector's partial over its k_lanes with a shuffle ladder:
          // simd_sum would mix the results_per_simdgroup rows a simdgroup spans.
          for (int v = 0; v < vecs_per_tg; v++) {
            if constexpr (k_lanes >= 32) {
              result[v] += simd_shuffle_down(result[v], 16);
            }
            if constexpr (k_lanes >= 16) {
              result[v] += simd_shuffle_down(result[v], 8);
            }
            if constexpr (k_lanes >= 8) {
              result[v] += simd_shuffle_down(result[v], 4);
            }
            if constexpr (k_lanes >= 4) {
              result[v] += simd_shuffle_down(result[v], 2);
            }
            if constexpr (k_lanes >= 2) {
              result[v] += simd_shuffle_down(result[v], 1);
            }
          }
        
        }

        // `qmv_wide_impl` for the ONE tile of a matrix with out_vec_size < 8
        // rows (tile_row0 == 0): the short tile splits K across k_folds slots
        // per row and sums the folds in ascending order, verbatim. Needs the
        // full threadgroup (2 simdgroups) and 8 * vecs_per_tg floats of
        // threadgroup memory. On return, `valid` lanes (k_lane == 0, slot <
        // out_vec_size) hold row `slot`'s totals in `result`.
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>
        METAL_FUNC void qmv_wide_reg_partial(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_vec_size,
            const int M,
            threadgroup float* fold_partials,
            uint simd_gid,
            uint simd_lid,
            thread float (&result)[vecs_per_tg],
            thread bool& valid,
            thread int& row_out) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
          constexpr int sub = 8; // values per sub-chunk (== bits bytes, byte-aligned)
          typedef float U;
          constexpr int rows_per_tg = results_per_simdgroup * num_simdgroups;
          const short k_lane = simd_lid % k_lanes;
          const short sg_row = simd_lid / k_lanes;
          const short slot = simd_gid * results_per_simdgroup + sg_row;
          const int tile_row0 = 0;
          const int tile_rows = min(out_vec_size - tile_row0, rows_per_tg);
          const int vec0 = 0;
          int k_folds = 1;
          int fold = 0;
          int out_row = tile_row0 + slot;
          if (tile_rows < rows_per_tg) {
            while (k_folds * 2 * tile_rows <= rows_per_tg) {
              k_folds *= 2;
            }
            fold = slot / tile_rows;
            out_row = tile_row0 + slot % tile_rows;
          }
          const int row = min(out_row, out_vec_size - 1);
        
          const int in_vec_size_w = in_vec_size * bits / 8; // bytes per weight row
          const int in_vec_size_g = in_vec_size / group_size;
          const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
          const device T* srow = scales + row * in_vec_size_g;
          const device T* brow = biases + row * in_vec_size_g;
        
          const device T* xv[vecs_per_tg];
          for (int v = 0; v < vecs_per_tg; v++) {
            xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
          }
        
          for (int v = 0; v < vecs_per_tg; v++) { result[v] = 0; }
        
          // Each lane reduces a strided subset of the row's groups: decode the group in
          // 8-value sub-chunks and reuse each chunk across the streamed vectors.
          const int g_stride = k_lanes * k_folds;
          const int g_first = fold < k_folds ? k_lane + fold * k_lanes : in_vec_size_g;
          // Group loop. The 4-bit path is unrolled by g_unroll: the scale, the bias
          // and the packed weights of every group a trip covers are loaded before any
          // of them is decoded, so g_unroll independent memory requests per thread are
          // in flight at once. At the decode output widths only 40-80 threadgroups are
          // resident on the whole GPU, so there is no other thread to hide a load
          // behind and the stock loop paid one memory round trip per group. Every
          // group is decoded by the same expression as before, every vector sums its
          // terms in ascending element order, and the group partials still reach
          // result[v] in ascending group order, so the arithmetic is bit identical.
          static_assert(bits == 4, "MLXFAST-HDRTRIM: 4-bit only");
          if constexpr (bits == 4) {
            constexpr int g_unroll = 3;
            constexpr int packs_per_group = group_size / sub;
            int g = g_first;
            for (; g + (g_unroll - 1) * g_stride < in_vec_size_g;
                 g += g_unroll * g_stride) {
              float su[g_unroll];
              float bu[g_unroll];
              uint32_t wpack[g_unroll][packs_per_group];
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                su[u] = static_cast<float>(srow[gu]);
                bu[u] = static_cast<float>(brow[gu]);
                const device uint32_t* wg =
                    (const device uint32_t*)(wrow + gu * (group_size * bits / 8));
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  wpack[u][sc] = wg[sc];
                }
              }
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                const float s = su[u];
                const float b = bu[u];
                const float s_hi = s / 16.0f;
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  const int k0 = gu * group_size + sc * sub;
                  const uint32_t p = wpack[u][sc];
                  U w_dq[sub];
        #pragma unroll
                  for (int i = 0; i < sub / 2; i++) {
                    const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                    w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                    w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                  }
                  // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                  // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                  // of eight, issued before any product. The element-major order below is
                  // unchanged, so every vector still sums its terms in ascending element
                  // order -- bit identical.
                  vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                    for (int c = 0; c < sub / 4; c++) {
                      xq[v][c] = xc4[c];
                    }
                  }
                  U accv[vecs_per_tg] = {0};
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                    }
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    result[v] += accv[v];
                  }
                }
              }
            }
            // Groups left over when the row's group count is not a multiple of
            // g_unroll, in the same ascending order.
            for (; g < in_vec_size_g; g += g_stride) {
              const float s = static_cast<float>(srow[g]);
              const float b = static_cast<float>(brow[g]);
              const float s_hi = s / 16.0f;
              const device uint32_t* wg =
                  (const device uint32_t*)(wrow + g * (group_size * bits / 8));
              uint32_t wpack[group_size / sub];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                wpack[sc] = wg[sc];
              }
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const uint32_t p = wpack[sc];
                U w_dq[sub];
        #pragma unroll
                for (int i = 0; i < sub / 2; i++) {
                  const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                  w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                  w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                }
                // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                // of eight, issued before any product. The element-major order below is
                // unchanged, so every vector still sums its terms in ascending element
                // order -- bit identical.
                vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
                    xq[v][c] = xc4[c];
                  }
                }
                U accv[vecs_per_tg] = {0};
        #pragma unroll
                for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                  }
                }
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  result[v] += accv[v];
                }
              }
            }
          }
          // Reduce each vector's partial over its k_lanes with a shuffle ladder:
          // simd_sum would mix the results_per_simdgroup rows a simdgroup spans.
          for (int v = 0; v < vecs_per_tg; v++) {
            if constexpr (k_lanes >= 32) {
              result[v] += simd_shuffle_down(result[v], 16);
            }
            if constexpr (k_lanes >= 16) {
              result[v] += simd_shuffle_down(result[v], 8);
            }
            if constexpr (k_lanes >= 8) {
              result[v] += simd_shuffle_down(result[v], 4);
            }
            if constexpr (k_lanes >= 4) {
              result[v] += simd_shuffle_down(result[v], 2);
            }
            if constexpr (k_lanes >= 2) {
              result[v] += simd_shuffle_down(result[v], 1);
            }
          }
        
          valid = false;
          row_out = out_row;
          if (k_folds > 1) {
            if (k_lane == 0) {
              for (int v = 0; v < vecs_per_tg; v++) {
                fold_partials[slot * vecs_per_tg + v] = result[v];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k_lane == 0 && slot < tile_rows) {
              for (int v = 0; v < vecs_per_tg; v++) {
                U total = fold_partials[slot * vecs_per_tg + v];
                for (int f = 1; f < k_folds; f++) {
                  total += fold_partials[(slot + f * tile_rows) * vecs_per_tg + v];
                }
                result[v] = total;
              }
              valid = true;
            }
            return;
          }
          if (k_lane == 0 && fold == 0 && out_row < out_vec_size) { valid = true; }
        }
        """#

    static let helpers4 = helpersCore4 + helpersMLX
}
