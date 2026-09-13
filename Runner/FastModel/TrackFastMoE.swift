// TrackFastMoE.swift -- routed-expert GEMVs over MLX's own quantized GEMV
// arithmetic, with the expert addressing done here.
//
// MLX's `gather_qmm` op costs ~40 us of fixed time per call at decode shapes
// regardless of bytes (measured 55 us for 10 experts x [640, 2560] against
// 11 us for a dense GEMV of the same bytes). The kernels below take the
// per-row functions of `mlx/backend/metal/kernels/quantized.h` VERBATIM
// (`qmv_fast_impl` for K % 512 == 0, `qmv_impl` otherwise -- the same choice
// the op makes) and only replace the batch addressing, so every output
// element is the same accumulation the op computes. Verified bit-exact
// against `SwitchGLU` at one- and multi-row windows.

import Foundation
import MLX

enum TrackFastMoEKernels {
    /// `mlx/backend/metal/kernels/quantized.h` lines 12-393 and 757-987.
    /// MLX `quantized.h` verbatim: the pack helpers, `load_vector*` and `qdot*` the replicas call.
    static let helpersCore = #"""
#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
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
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
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

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
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
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

"""#

    /// MLX `quantized.h` verbatim: the full `qmv_fast_impl` / `qmv_impl` kernels (gather gate|up / single launches).
    static let helpersMLX = #"""
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

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
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits, bool EXACT_TAIL = false>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
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
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    // MLXFAST-FULLTAIL. Same argument as qmv_reg: an exact K makes remaining
    // 0 or values_per_thread, so the unchecked helpers are bit-identical.
    if constexpr (EXACT_TAIL) {
      if (remaining > 0) {
        U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
        for (int row = 0;
             row < results_per_simdgroup && out_row + row < out_vec_size;
             row++) {
          auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
          const device T* sl = scales + row * in_vec_size_g;
          const device T* bl = biases + row * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result[row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
        }
      }
    } else if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if constexpr (EXACT_TAIL) {
      if (remaining > 0) {
        U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
          auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
          const device T* sl = scales + row * in_vec_size_g;
          const device T* bl = biases + row * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result[row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
        }
      }
    } else if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

// Affine analog of fp_qmv_wide. Weights carry a scale and bias per group, so
// each group is decoded in 8-value sub-chunks (scale * q + bias, registers
// bounded for any group_size) and reused across the vecs_per_tg vectors.

"""#

    static let helpers = helpersCore + helpersMLX

    /// gate and up in one launch: y-blocks [0, NB) compute gate rows, [NB, 2NB) up rows,
    /// each threadgroup handling RB consecutive 8-row blocks. grid threads (32, NB*2*2, B), threadgroup (32, 2, 1),
    /// NB = ceil(N / (8*RB)).
    ///   wg/sg/bg, wu/su/bu: [E, N, K/8] / [E, N, K/32]; x [R, K]; idx [B] uint32 expert; xrow [B] uint32 row
    ///   -> gate [B, N], up [B, N]
    static let gateUpSource = """
        const uint b = threadgroup_position_in_grid.z;
        const uint e = idx[b];
        const uint r = xrow[b];
        const uint kw = (uint)K / 8;
        const uint kg = (uint)K / GS;
        const uint nblocks = (uint)N / 8;
        const uint NB = (nblocks + RB - 1) / RB;
        const uint yb = threadgroup_position_in_grid.y;
        const bool isUp = yb >= NB;
        const uint blk0 = (isUp ? yb - NB : yb) * RB;
        const device T* xb = x + (size_t)r * (size_t)K;
        const device uint32_t* wb = (isUp ? wu : wg) + (size_t)e * (size_t)N * kw;
        const device T* sb = (isUp ? su : sg) + (size_t)e * (size_t)N * kg;
        const device T* bb = (isUp ? bu : bg) + (size_t)e * (size_t)N * kg;
        device T* yb_ = (isUp ? up : gate) + (size_t)b * (size_t)N;
        for (uint i = 0; i < RB; ++i) {
            const uint blk = blk0 + i;
            if (blk >= nblocks) break;
            uint3 tid = uint3(0, blk, 0);
            if (FAST) { qmv_fast_impl<T, GS, BITS>(wb, sb, bb, xb, yb_, K, N, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup); }
            else { qmv_impl<T, GS, BITS, EXACT_TAIL>(wb, sb, bb, xb, yb_, K, N, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup); }
        }
        """

    /// one projection per pair (the down projection), RB 8-row blocks per threadgroup.
    /// x [B, K] (row b), out [B, N]. grid threads (32, NB*2, B).
    static let singleSource = """
        const uint b = threadgroup_position_in_grid.z;
        const uint e = idx[b];
        const uint kw = (uint)K / 8;
        const uint kg = (uint)K / GS;
        const uint nblocks = (uint)N / 8;
        const uint blk0 = threadgroup_position_in_grid.y * RB;
        const device T* xb = x + (size_t)b * (size_t)K;
        const device uint32_t* wb = w + (size_t)e * (size_t)N * kw;
        const device T* sb = scales + (size_t)e * (size_t)N * kg;
        const device T* bb = biases + (size_t)e * (size_t)N * kg;
        device T* yb_ = out + (size_t)b * (size_t)N;
        for (uint i = 0; i < RB; ++i) {
            const uint blk = blk0 + i;
            if (blk >= nblocks) break;
            uint3 tid = uint3(0, blk, 0);
            if (FAST) { qmv_fast_impl<T, GS, BITS>(wb, sb, bb, xb, yb_, K, N, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup); }
            else { qmv_impl<T, GS, BITS, EXACT_TAIL>(wb, sb, bb, xb, yb_, K, N, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup); }
        }
        """

    nonisolated(unsafe) static let gateUpKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "x", "idx", "xrow"],
        outputNames: ["gate", "up"],
        source: gateUpSource, header: helpers, ensureRowContiguous: true)

    nonisolated(unsafe) static let singleKernel = MLXFast.metalKernel(
        name: "track_moe_single",
        inputNames: ["w", "scales", "biases", "x", "idx"],
        outputNames: ["out"],
        source: singleSource, header: helpers, ensureRowContiguous: true)

    static func isFast(k: Int, n: Int) -> Bool { n % 8 == 0 && k % 512 == 0 }

    /// `TRACK_QMV_FULLTAIL`: skip per-element tail bounds checks when K is an
    /// exact multiple of the qmv lane coverage. Default ON. `"0"` restores the
    /// vendored `_safe` tail.
    static var fulltailEnabled: Bool {
        (ProcessInfo.processInfo.environment["TRACK_QMV_FULLTAIL"] ?? "1") != "0"
    }

    /// MLX `get_pack_factor<bits, 32>()`: values packed in one 32-bit word.
    static func packFactor(bits: Int, wsize: Int = 32) -> Int {
        if bits == 3 || bits == 5 { return 8 }
        if bits == 6 { return 4 }
        return wsize / bits
    }

    /// `qmv_impl` / inject: `values_per_thread = pack_factor * packs_per_thread`.
    static func valuesPerThread(bits: Int, packsPerThread: Int = 1) -> Int {
        packFactor(bits: bits) * packsPerThread
    }

    /// Tail coverage is exact: `remaining` is 0 or `values_per_thread`, never a
    /// partial slice. `qmv_impl` `packs_per_thread` is 1.
    static func isExactTail(k: Int, bits: Int, packsPerThread: Int = 1) -> Bool {
        let vpt = valuesPerThread(bits: bits, packsPerThread: packsPerThread)
        return vpt > 0 && k % vpt == 0
    }

    static func useExactTail(k: Int, bits: Int, packsPerThread: Int = 1) -> Bool {
        fulltailEnabled && isExactTail(k: k, bits: bits, packsPerThread: packsPerThread)
    }

    /// 8-row blocks per threadgroup: the K=640 down projection is overhead-bound
    /// at one block per threadgroup.
    static func rowBlocks(k: Int) -> Int { k % 512 == 0 ? 1 : 4 }

    /// gate/up for `B` (row, expert) pairs.
    static func gateUp(
        wg: MLXArray, sg: MLXArray, bg: MLXArray, wu: MLXArray, su: MLXArray, bu: MLXArray,
        x: MLXArray, idx: MLXArray, xrow: MLXArray, groupSize: Int, bits: Int, rowBlocks: Int? = nil
    ) -> (gate: MLXArray, up: MLXArray) {
        let B = idx.dim(0), K = x.dim(1), N = wg.dim(1)
        precondition(N % 8 == 0 && bits == 4)
        let rb = rowBlocks ?? Self.rowBlocks(k: K)
        let nb = (N / 8 + rb - 1) / rb
        let outs = gateUpKernel(
            [wg, sg, bg, wu, su, bu, x, idx, xrow],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("FAST", isFast(k: K, n: N)), ("RB", rb), ("K", K), ("N", N), ("EXACT_TAIL", useExactTail(k: K, bits: bits))],
            grid: (32, nb * 2 * 2, B), threadGroup: (32, 2, 1),
            outputShapes: [[B, N], [B, N]], outputDTypes: [x.dtype, x.dtype])
        return (outs[0], outs[1])
    }

    static func single(
        w: MLXArray, scales: MLXArray, biases: MLXArray, x: MLXArray, idx: MLXArray,
        groupSize: Int, bits: Int, rowBlocks: Int? = nil
    ) -> MLXArray {
        let B = idx.dim(0), K = x.dim(1), N = w.dim(1)
        precondition(N % 8 == 0 && bits == 4)
        let rb = rowBlocks ?? Self.rowBlocks(k: K)
        let nb = (N / 8 + rb - 1) / rb
        return singleKernel(
            [w, scales, biases, x, idx],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("FAST", isFast(k: K, n: N)), ("RB", rb), ("K", K), ("N", N), ("EXACT_TAIL", useExactTail(k: K, bits: bits))],
            grid: (32, nb * 2, B), threadGroup: (32, 2, 1),
            outputShapes: [[B, N]], outputDTypes: [x.dtype])[0]
    }
}

// MARK: router tail: top-k + softmax + scatter in ONE launch
//
// Replaces `argPartition(-logits, kth: k-1)[..., ..<k]`, `takeAlong`,
// `softmax(precise: true)` and the index store. MLX's GPU argpartition is a
// full stable merge sort (`sort.h`: strict `<` in the thread sort and the
// merge), so the first k entries are the k largest logits in descending
// value order with ties in ascending index order.
//
// Top-k is K sequential ArgMax reductions. Each reduction is the same
// order-fixed simd_shuffle_down tree as `arg_reduce.metal` (offset
// 16,8,4,2,1; compare is ArgMax: higher value, then lower index). No
// `simd_max`/`simd_min` and no atomics: the winner is independent of
// unspecified reduction order. Lane-local scan uses strict `>` over
// strided experts `e = lane + 32*j`, so a within-lane tie keeps the
// lower index. Softmax over the K selected logits stays
// `softmax_single_row` verbatim (N_READS = 4, one simdgroup, `fast::exp`,
// per-thread sequential partials, `simd_sum`, reciprocal) so the weights
// match `softmax(..., precise: true)` bit-for-bit. Scatter writes idx
// uint32 and w float32 in that same descending/tie order.
//
// `TRACK_ROUTER_FUSED=0` restores the three-op chain. Shapes outside the
// warmed set (see `fusedRouterEnabled` call sites) also fall back.

extension TrackFastMoEKernels {
    /// `TRACK_ROUTER_FUSED=0` restores `argPartition` + `takeAlong` +
    /// `softmax(precise: true)`. Default ON.
    static let fusedRouterEnabled =
        (ProcessInfo.processInfo.environment["TRACK_ROUTER_FUSED"] ?? "1") != "0"

    /// The eager chain the fused kernel is bit-exact against.
    static func routeChain(logits: MLXArray, topK: Int) -> (idx: MLXArray, w: MLXArray) {
        let idx = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let w = softmax(takeAlong(logits, idx, axis: -1), axis: -1, precise: true)
        return (idx, w)
    }

    /// logits f32 [R, E] -> idx uint32 [R, K], w f32 [R, K].
    static let routeSource = """
        constexpr int E_PER = (E + 31) / 32;
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        // Shared-expert gate (1 row, K = KD): one token routes to `qmv`'s small-N
        // branch (simdgroup 0), two to eight to `qmv_wide`'s short tile, which
        // folds K across the 8 slots of BOTH simdgroups.
        if constexpr (HAS_GATE) {
            const device T* xr = x + (size_t)row * (size_t)KD;
            if constexpr (VPT == 1) {
                track_inject_qmv<T, GS, BITS, KD, 1, 4, EXACT_TAIL>(wg, sgw, bgw, xr, gate + row, sg, lane);
            } else {
                threadgroup float fp[8];
                float r[1]; bool valid = false; int orow = 0;
                qmv_wide_reg_partial<T, GS, BITS, 1, 8>(wg, sgw, bgw, xr, KD, 1, 1, fp, sg, lane, r, valid, orow);
                if (valid) { gate[row] = static_cast<T>(r[0]); }
            }
        }
        threadgroup float selv[K];
        threadgroup uint seli[K];
        // MLXFAST-ROUTESG1: for one token the shared-gate GEMV above runs on
        // simdgroup 0 alone (`track_inject_qmv` returns at once on simdgroup 1),
        // and the top-K walk used to queue behind it on the same simdgroup. Run
        // the walk on simdgroup 1 instead so the two latency chains overlap; the
        // walk's arithmetic, tie rule and the softmax below are untouched. Wide
        // windows keep the gate on both simdgroups and the walk on simdgroup 0.
        constexpr uint SEL_SG = (VPT == 1) ? 1u : 0u;
        if (sg == SEL_SG) {
        const device float* lr = logits + (size_t)row * (size_t)E;
        // each lane owns E_PER experts: e = lane + 32 * j (strided so a tie at
        // the same value resolves to the lowest index across lanes too)
        float v[E_PER];
        bool taken[E_PER];
        for (int j = 0; j < E_PER; ++j) {
            const int e = (int)lane + 32 * j;
            v[j] = (e < E) ? lr[e] : -INFINITY;
            taken[j] = (e >= E);
        }
        for (int k = 0; k < K; ++k) {
            // lane-local best: largest value, then lowest index (strict `>`
            // keeps the earlier, lower-index slot on a tie).
            float bv = -INFINITY; int bj = -1;
            for (int j = 0; j < E_PER; ++j) {
                if (!taken[j] && (v[j] > bv)) { bv = v[j]; bj = j; }
            }
            uint bi = (bj >= 0) ? (uint)(lane + 32 * bj) : 0xffffffffu;
            const uint local_i = bi;
            // arg_reduce.metal ArgMax tree: shuffle_down offsets 16..1,
            // take neighbor if it has a strictly larger value or the same
            // value and a smaller index. Lane 0 holds the winner.
            for (ushort off = 16; off > 0; off >>= 1) {
                const float nv = simd_shuffle_down(bv, off);
                const uint ni = simd_shuffle_down(bi, off);
                if (nv > bv || (nv == bv && ni < bi)) { bv = nv; bi = ni; }
            }
            const uint gidx = simd_shuffle(bi, (ushort)0);
            if (lane == 0) { selv[k] = bv; seli[k] = bi; }
            if (gidx == local_i && bj >= 0) { taken[bj] = true; }
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg != 0) { return; }
        // softmax_single_row over the K selected logits (AccT = float)
        constexpr int N_READS = 4;
        float ld[N_READS];
        for (int i = 0; i < N_READS; i++) {
            const int p = (int)lane * N_READS + i;
            ld[i] = (p < K) ? selv[p] : -INFINITY;
        }
        float maxval = -FLT_MAX;
        for (int i = 0; i < N_READS; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
        maxval = simd_max(maxval);
        float normalizer = 0;
        for (int i = 0; i < N_READS; i++) {
            float exp_x = fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        normalizer = 1 / normalizer;
        for (int i = 0; i < N_READS; i++) {
            const int p = (int)lane * N_READS + i;
            if (p < K) {
                w[(size_t)row * K + p] = ld[i] * normalizer;
                idx[(size_t)row * K + p] = seli[p];
            }
        }
        """

    nonisolated(unsafe) static let routeKernel = MLXFast.metalKernel(
        name: "track_moe_route",
        inputNames: ["logits", "x", "wg", "sgw", "bgw"],
        outputNames: ["idx", "w", "gate"],
        source: routeSource, header: TrackFastKernels.mixerHeadHeader + wideHelpers, ensureRowContiguous: true)
    /// One-token instantiation: same source, wide bodies replaced by their declarations.
    nonisolated(unsafe) static let routeKernel1 = MLXFast.metalKernel(
        name: "track_moe_route_1",
        inputNames: ["logits", "x", "wg", "sgw", "bgw"],
        outputNames: ["idx", "w", "gate"],
        source: routeSource, header: TrackFastKernels.mixerHeadHeader + wideDecls, ensureRowContiguous: true)

    /// logits f32 [..., E] -> (idx uint32 [..., K], w f32 [..., K])
    /// Also the shared-expert gate logit per row: x [..., KD] x sharedGate [1, KD] -> gate [...] (pre-sigmoid).
    static func route(logits: MLXArray, x: MLXArray, sharedGate: TrackQuantWeight?, topK: Int)
        -> (idx: MLXArray, w: MLXArray, gate: MLXArray)
    {
        precondition(logits.dtype == .float32 && (sharedGate == nil || (sharedGate!.rows == 1 && sharedGate!.bits == 4)))
        let g = sharedGate
        let E = logits.dim(-1), KD = x.dim(-1)
        let lead = Array(logits.shape.dropLast())
        let R = lead.reduce(1, *)
        precondition(topK <= 32 && topK <= E && R >= 1 && (g == nil || R <= 8) && KD % 256 == 0)
        let simdgroups = g == nil ? 1 : 2
        let outs = (R == 1 ? routeKernel1 : routeKernel)(
            [logits.reshaped(R, E), x.reshaped(R, KD), g?.weight ?? x, g?.scales ?? x, g?.biases ?? x],
            template: [("E", E), ("K", topK), ("T", x.dtype), ("GS", g?.groupSize ?? 32), ("BITS", g?.bits ?? 4), ("KD", KD), ("VPT", R), ("HAS_GATE", g != nil), ("EXACT_TAIL", useExactTail(k: KD, bits: g?.bits ?? 4))],
            grid: (32, R * simdgroups, 1), threadGroup: (32, simdgroups, 1),
            outputShapes: [[R, topK], [R, topK], [R]], outputDTypes: [.uint32, .float32, x.dtype])
        return (outs[0].reshaped(lead + [topK]), outs[1].reshaped(lead + [topK]), outs[2].reshaped(lead))
    }
}

// MARK: depth-1 next-K-block weight staging (decode expert GEMVs)
//
// Presence check (w67):
//   (a) Production `qdot` 4-bit reads `uint16_t` words from device, not 32-bit
//       packed loads (roster row 22 / c7352109). Byte-at-a-time is also absent
//       on this path.
//   (b) Served `qmv_fast_reg` / `qmv_fast_reg_dual` / `qmv_reg` issue those
//       loads inside the current block's qdot. `qmv_fast_reg_pf*` below is an
//       unused experiment, not dispatched.
// Staging prefetches the next K-block's uint16 weight words into registers
// and runs the same qdot expression on the staged set. Packed 32-bit loads
// stay a separate job. No second threadgroup buffer.
// `TRACK_WEIGHT_STAGING=0` restores the original kernel objects.

enum TrackWeightStaging {
    static func resolves(_ raw: String?) -> Bool {
        (raw ?? "1") != "0"
    }

    static let enabled = resolves(
        ProcessInfo.processInfo.environment["TRACK_WEIGHT_STAGING"])

    /// Current decode gate/up reuse geometry: 2 rows/simdgroup, 2 simdgroups.
    static let gateUpRowsPerSimdgroup = 2
    /// Current decode down geometry: 4 output rows per threadgroup.
    static let downRowsPerThreadgroup = 4

    static func stagesGateUp(
        s: Int, dtype: DType, kd: Int, n: Int, groupSize: Int, bits: Int,
        sharedAffine: Bool, enabled: Bool
    ) -> Bool {
        enabled
            && dtype == .bfloat16 && s == 1 && kd == 2560 && n == 640
            && groupSize == 32 && bits == 4 && sharedAffine
    }

    static func stagesDown(
        windowTokens: Int, f: Int, h: Int, bits: Int, enabled: Bool
    ) -> Bool {
        enabled && windowTokens == 1 && f == 640 && h == 2560 && bits == 4
    }

    enum GateUpDispatch: Equatable {
        case reuseStaged
        case reuse
        case act
    }

    static func gateUpDispatch(
        s: Int, dtype: DType, kd: Int, n: Int, groupSize: Int, bits: Int,
        sharedAffine: Bool, enabled: Bool
    ) -> GateUpDispatch {
        let reuse = dtype == .bfloat16 && s == 1 && kd == 2560 && n == 640
            && groupSize == 32 && bits == 4 && sharedAffine
        guard reuse else { return .act }
        return stagesGateUp(
            s: s, dtype: dtype, kd: kd, n: n, groupSize: groupSize, bits: bits,
            sharedAffine: sharedAffine, enabled: enabled) ? .reuseStaged : .reuse
    }

    enum DownDispatch: Equatable {
        case combineStaged
        case combine
    }

    static func downDispatch(
        windowTokens: Int, f: Int, h: Int, bits: Int, enabled: Bool
    ) -> DownDispatch {
        stagesDown(
            windowTokens: windowTokens, f: f, h: h, bits: bits, enabled: enabled)
            ? .combineStaged : .combine
    }
}

// MARK: fused routed-expert MLP: gate|up GEMVs + SwiGLU in one launch, down GEMV
//       + expert-weighted combine in one launch. The GEMV walks are MLX's own
//       `qmv_fast_impl` / `qmv_impl` (normal branch) with the result kept in
//       registers instead of stored: same lanes, same accumulation, same
//       `simd_sum`, same bf16 rounding of each expert's output before the
//       epilogue arithmetic (`static_cast<T>`), which is what the separate
//       launches did. `TRACK_SWIGLU_EPILOGUE=0` restores those two launches.

extension TrackFastMoEKernels {
    static let regHelpers = #"""

        // qmv_fast_impl with `out_row` given and the row results returned
        // (all lanes hold them after simd_sum). x points at the vector.
        // MLXFAST-MIX2ROW: only the number of independent contiguous rows varies.
        // The default preserves every existing four-row caller's arithmetic.
        template <typename T, int group_size, int bits, int results_per_simdgroup = 4>
        METAL_FUNC void qmv_fast_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[results_per_simdgroup]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
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

        // Same walk as qmv_fast_reg above, with in_vec_size a template so the
        // K loop bound is compile-time. Call sites that pass K as a function
        // argument keep the original overload; mixer staging instantiates this
        // one. Same lanes, same order, same qdot.
        template <typename T, int group_size, int bits, int in_vec_size, int results_per_simdgroup>
        METAL_FUNC void qmv_fast_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int out_row,
            uint simd_lid,
            thread float (&result)[results_per_simdgroup]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) { result[row] = 0; }
          constexpr int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          constexpr int in_vec_size_g = in_vec_size / group_size;
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
        // MLXFAST-FULLTAIL: EXACT_TAIL says the caller's `in_vec_size` is a
        // multiple of values_per_thread. See the tail block below for what that
        // buys and why it stays bit-identical.
        // MLXFAST-DOWNRPS: NR contiguous rows per simdgroup; each row's walk,
        // accumulation order and simd_sum are unchanged for any NR.
        template <typename T, int group_size, int bits, bool EXACT_TAIL = false, int NR = 4>
        METAL_FUNC void qmv_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[NR]) {
          constexpr int results_per_simdgroup = NR;
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
          // MLXFAST-FULLTAIL. When in_vec_size is a multiple of
          // values_per_thread, `remaining` is provably 0 or values_per_thread and
          // never a partial slice: in_vec_size - k is a multiple of
          // values_per_thread (k advances by block_size = 32 * values_per_thread)
          // and so is simd_lid * values_per_thread, so their difference is too,
          // and the clamp leaves only the two endpoints. The _safe helpers then
          // run the SAME arithmetic in the SAME order as the plain ones -- their
          // bodies are identical with `N` in place of `values_per_thread` -- but
          // over a RUNTIME trip count. That keeps x_thread dynamically indexed,
          // which pins the array in thread-local scratch for the whole function
          // instead of registers, and costs the main loop as well as the tail.
          // K = 640 (down) is 2.5 blocks, so a fifth of that GEMV's work sits in
          // this branch; K = 2560 puts a tenth there. Both are exact multiples of
          // 8, so EXACT_TAIL erases the runtime-indexed code path entirely.
          // Bit-identical by construction, not by tolerance.
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
              U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device T* sl = scales + row * in_vec_size_g;
                const device T* bl = biases + row * in_vec_size_g;
                U s = sl[0];
                U b = bl[0];
                result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
              }
            }
          } else if (remaining > 0) {
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
        // EXACT_TAIL as in qmv_reg above: the caller's in_vec_size is a multiple
        // of values_per_thread, so the runtime-indexed tail can be compiled away.
        template <typename T, int group_size, int bits, bool SILU, bool EXACT_TAIL = false>
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
          // MLXFAST-FULLTAIL, same argument as qmv_reg.
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
              U sum = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x, x_thread)
                           : load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
                U s = sr[row][0];
                U b = br[row][0];
                result[row] += qdot<U, values_per_thread, bits>(wr[row], x_thread, s, b, sum);
              }
            }
          } else if (remaining > 0) {
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

        // Same walk as qmv_reg_rows above, with in_vec_size a template so the
        // K loop bound is compile-time. Mixer staging instantiates this; the
        // original overload stays for every runtime-K caller.
        template <typename T, int group_size, int bits, int in_vec_size, bool SILU, bool EXACT_TAIL = false>
        METAL_FUNC void qmv_reg_rows(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
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
          constexpr int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          constexpr int in_vec_size_g = in_vec_size / group_size;
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
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
              U sum = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x, x_thread)
                           : load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
                U s = sr[row][0];
                U b = br[row][0];
                result[row] += qdot<U, values_per_thread, bits>(wr[row], x_thread, s, b, sum);
              }
            }
          } else if (remaining > 0) {
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

    /// Routed experts, gate|up + SwiGLU. x [S, KD], idx/xrow uint32 [BR] ->
    /// act [BR, N]. grid threads (32, N/8, BR), threadgroup (32, 2, 1).
    static let gateUpActSource = """
        const uint z = threadgroup_position_in_grid.z;
        if (z == (uint)BR) {
            // Shared expert: gate rows [0, N) and up rows [N, 2N) of the fused
            // shared matrix, all S tokens. One token routes to `qmv_fast`, two to
            // eight to `qmv_wide` (full tiles), exactly as the separate launches.
            const uint kw2 = (uint)KD / 8;
            const uint kg2 = (uint)KD / GS;
            const int tile = (int)threadgroup_position_in_grid.y;
            if constexpr (VPT == 1) {
                const int out_row = tile * 8 + (int)simdgroup_index_in_threadgroup * 4;
                float g[4], u[4];
                qmv_fast_reg<T, GS, BITS>(wsh, ssh, bsh, x, KD, out_row, thread_index_in_simdgroup, g);
                qmv_fast_reg<T, GS, BITS>(wsh + (size_t)N * kw2, ssh + (size_t)N * kg2, bsh + (size_t)N * kg2, x, KD, out_row, thread_index_in_simdgroup, u);
                if (thread_index_in_simdgroup == 0) {
                    for (int i = 0; i < 4; ++i) {
                        act[(size_t)BR * (size_t)N + (size_t)(out_row + i)] = mlx_silu(static_cast<T>(g[i])) * static_cast<T>(u[i]);
                    }
                }
            } else {
                const int row = tile * 8 + (int)simdgroup_index_in_threadgroup * 4 + (int)(thread_index_in_simdgroup / 8);
                float g[VPT], u[VPT];
                qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wsh, ssh, bsh, x, KD, VPT, row, thread_index_in_simdgroup, g);
                qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wsh + (size_t)N * kw2, ssh + (size_t)N * kg2, bsh + (size_t)N * kg2, x, KD, VPT, row, thread_index_in_simdgroup, u);
                if ((thread_index_in_simdgroup % 8) == 0) {
                    for (int v = 0; v < VPT; ++v) {
                        act[(size_t)(BR + v) * (size_t)N + (size_t)row] = mlx_silu(static_cast<T>(g[v])) * static_cast<T>(u[v]);
                    }
                }
            }
            return;
        }
        const uint e = idx[z];
        const uint r = xrow[z];
        const uint kw = (uint)KD / 8;
        const uint kg = (uint)KD / GS;
        const int out_row = (int)threadgroup_position_in_grid.y * 8 + (int)simdgroup_index_in_threadgroup * 4;
        const device T* xb = x + (size_t)r * (size_t)KD;
        const size_t pitch = (size_t)N * (size_t)TILES;
        const size_t eoff_g = (size_t)e * pitch;
        const size_t eoff_u = TILES > 1 ? eoff_g + (size_t)N : (size_t)e * (size_t)N;
        float g[4], u[4];
        if (FAST) {
            qmv_fast_reg<T, GS, BITS>(wg + eoff_g * kw, sg + eoff_g * kg, bg + eoff_g * kg, xb, KD, out_row, thread_index_in_simdgroup, g);
            qmv_fast_reg<T, GS, BITS>(wu + eoff_u * kw, su + eoff_u * kg, bu + eoff_u * kg, xb, KD, out_row, thread_index_in_simdgroup, u);
        } else {
            qmv_reg<T, GS, BITS, EXACT_TAIL>(wg + eoff_g * kw, sg + eoff_g * kg, bg + eoff_g * kg, xb, KD, out_row, thread_index_in_simdgroup, g);
            qmv_reg<T, GS, BITS, EXACT_TAIL>(wu + eoff_u * kw, su + eoff_u * kg, bu + eoff_u * kg, xb, KD, out_row, thread_index_in_simdgroup, u);
        }
        if (thread_index_in_simdgroup == 0) {
            for (int i = 0; i < 4; ++i) {
                const T gv = static_cast<T>(g[i]);
                const T uv = static_cast<T>(u[i]);
                act[(size_t)z * (size_t)N + (size_t)(out_row + i)] = mlx_silu(gv) * uv;
            }
        }
        """

    nonisolated(unsafe) static let gateUpActKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up_act",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
        outputNames: ["act"],
        source: gateUpActSource, header: helpersCore + TrackFastKernels.exactHeader + regHelpers + wideHelpers,
        ensureRowContiguous: true)
    nonisolated(unsafe) static let gateUpActKernel1 = MLXFast.metalKernel(
        name: "track_moe_gate_up_act_1",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
        outputNames: ["act"],
        source: gateUpActSource, header: helpersCore + TrackFastKernels.exactHeader + regHelpers + wideDecls,
        ensureRowContiguous: true)

    /// PB-556 / mixer-RPS family: one-token gate/up row ownership.
    /// Default ON: 1 row/simdgroup (was 2). Threadgroup stays 2 simdgroups, so
    /// rows/threadgroup go 4 → 2 and independently schedulable threadgroups
    /// double at N=640 (160 → 320 per expert slot). `TRACK_GATEUP_RPS=0`
    /// restores 2. S>1 and non-served shapes keep the 8-row fallback launch.
    static let gateUpReuseRpsEnabled =
        (ProcessInfo.processInfo.environment["TRACK_GATEUP_RPS"] ?? "1") != "0"
    static let gateUpReuseSimdgroupsPerThreadgroup = 2

    static func gateUpReuseRowsPerSimdgroup(enabled: Bool) -> Int {
        enabled ? 1 : 2
    }

    static var gateUpReuseRowsPerSimdgroup: Int {
        gateUpReuseRowsPerSimdgroup(enabled: gateUpReuseRpsEnabled)
    }

    static func gateUpReuseEligible(
        s: Int, dtype: DType, kd: Int, n: Int, groupSize: Int, bits: Int, sharedAffine: Bool
    ) -> Bool {
        dtype == .bfloat16 && s == 1 && kd == 2560 && n == 640
            && groupSize == 32 && bits == 4 && sharedAffine
    }

    static func gateUpReuseLaunch(
        n: Int, rowsPerSimdgroup: Int, slots: Int
    ) -> (
        grid: (Int, Int, Int), threadGroup: (Int, Int, Int), threadgroups: Int,
        rowsPerThreadgroup: Int
    ) {
        let tgY = gateUpReuseSimdgroupsPerThreadgroup
        let rowsPerThreadgroup = tgY * rowsPerSimdgroup
        let gridY = n / rowsPerSimdgroup
        precondition(rowsPerSimdgroup == 1 || rowsPerSimdgroup == 2)
        precondition(n % rowsPerThreadgroup == 0 && gridY % tgY == 0)
        return (
            (32, gridY, slots),
            (32, tgY, 1),
            (gridY / tgY) * slots,
            rowsPerThreadgroup
        )
    }

    static let gateUpReuseHelpers = #"""
        template <typename T, int group_size, int bits, int rows>
        METAL_FUNC void qmv_fast_reg_dual(
            const device uint32_t* w0,
            const device T* scales0,
            const device T* biases0,
            const device uint32_t* w1,
            const device T* scales1,
            const device T* biases1,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result0)[rows],
            thread float (&result1)[rows]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws0 = (const device uint8_t*)w0;
          const device uint8_t* ws1 = (const device uint8_t*)w1;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < rows; row++) {
            result0[row] = 0;
            result1[row] = 0;
          }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws0 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          ws1 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          scales1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          for (int k = 0; k < in_vec_size; k += block_size) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < rows; row++) {
              auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
              const device T* sl0 = scales0 + row * in_vec_size_g;
              const device T* bl0 = biases0 + row * in_vec_size_g;
              U s0 = sl0[0];
              U b0 = bl0[0];
              result0[row] += qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
              auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
              const device T* sl1 = scales1 + row * in_vec_size_g;
              const device T* bl1 = biases1 + row * in_vec_size_g;
              U s1 = sl1[0];
              U b1 = bl1[0];
              result1[row] += qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
            }
            ws0 += block_size * bytes_per_pack / pack_factor;
            ws1 += block_size * bytes_per_pack / pack_factor;
            scales0 += block_size / group_size;
            scales1 += block_size / group_size;
            biases0 += block_size / group_size;
            biases1 += block_size / group_size;
            x += block_size;
          }
          for (int row = 0; row < rows; row++) {
            result0[row] = simd_sum(result0[row]);
            result1[row] = simd_sum(result1[row]);
          }
        }
        """#

    /// Next-K-block uint16 weight-word prefetch. Same 4-bit qdot body as
    /// `qdot` in helpersCore; loads happen one block ahead into registers.
    static let weightStagingHelpers = #"""
        template <typename U, int nwords>
        METAL_FUNC U qdot4_from_words(
            const thread uint16_t* ws,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          U accum = 0;
          for (int i = 0; i < nwords; i++) {
            accum +=
                (x_thread[4 * i] * (ws[i] & 0x000f) +
                 x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                 x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                 x_thread[4 * i + 3] * (ws[i] & 0xf000));
          }
          return scale * accum + sum * bias;
        }

        template <int rows, int nwords>
        METAL_FUNC void track_ws_fetch_rows(
            thread uint16_t* dst,
            const device uint8_t* ws,
            int in_vec_size_w) {
          for (int row = 0; row < rows; row++) {
            const device uint16_t* wl =
                (const device uint16_t*)(ws + row * in_vec_size_w);
            for (int i = 0; i < nwords; i++) {
              dst[row * nwords + i] = wl[i];
            }
          }
        }

        // Decode gate/up: K % 512 == 0, 2 rows/simdgroup. Prefetch the next
        // block's uint16 words (the same loads `qdot` 4-bit issues) before this
        // block's dequant. Activations, scales and biases stay on the original
        // schedule — weight stream only.
        template <typename T, int group_size, int bits, int rows>
        METAL_FUNC void qmv_fast_reg_dual_ws(
            const device uint32_t* w0,
            const device T* scales0,
            const device T* biases0,
            const device uint32_t* w1,
            const device T* scales1,
            const device T* biases1,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result0)[rows],
            thread float (&result1)[rows]) {
          static_assert(bits == 4, "weight staging: 4-bit");
          static_assert(rows == 2, "weight staging gated to 2 rows/simdgroup");
          constexpr int packs_per_thread = 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          constexpr int nwords = values_per_thread / 4;
          constexpr int WSTEP = block_size * bytes_per_pack / pack_factor;
          constexpr int GSTEP = block_size / group_size;
          const device uint8_t* ws0 = (const device uint8_t*)w0;
          const device uint8_t* ws1 = (const device uint8_t*)w1;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < rows; row++) {
            result0[row] = 0;
            result1[row] = 0;
          }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws0 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          ws1 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          scales1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          uint16_t ga[rows * nwords], ua[rows * nwords];
          uint16_t gb[rows * nwords], ub[rows * nwords];
          #define TRACK_WS_FETCH(GW, UW, OFFB) \
            track_ws_fetch_rows<rows, nwords>(GW, ws0 + (OFFB) * WSTEP, in_vec_size_w); \
            track_ws_fetch_rows<rows, nwords>(UW, ws1 + (OFFB) * WSTEP, in_vec_size_w);
          #define TRACK_WS_COMPUTE(GW, UW, OFFB) { \
            U sum = load_vector<T, U, values_per_thread, bits>( \
                x + (OFFB) * block_size, x_thread); \
            const device T* sc0 = scales0 + (OFFB) * GSTEP; \
            const device T* sc1 = scales1 + (OFFB) * GSTEP; \
            const device T* bi0 = biases0 + (OFFB) * GSTEP; \
            const device T* bi1 = biases1 + (OFFB) * GSTEP; \
            for (int row = 0; row < rows; row++) { \
              U s0 = sc0[row * in_vec_size_g]; \
              U b0 = bi0[row * in_vec_size_g]; \
              result0[row] += qdot4_from_words<U, nwords>( \
                  GW + row * nwords, x_thread, s0, b0, sum); \
              U s1 = sc1[row * in_vec_size_g]; \
              U b1 = bi1[row * in_vec_size_g]; \
              result1[row] += qdot4_from_words<U, nwords>( \
                  UW + row * nwords, x_thread, s1, b1, sum); \
            } \
          }
          const int nblocks = in_vec_size / block_size;
          TRACK_WS_FETCH(ga, ua, 0)
          int blk = 0;
          for (; blk + 1 < nblocks; blk += 2) {
            TRACK_WS_FETCH(gb, ub, blk + 1)
            TRACK_WS_COMPUTE(ga, ua, blk)
            if (blk + 2 < nblocks) { TRACK_WS_FETCH(ga, ua, blk + 2) }
            TRACK_WS_COMPUTE(gb, ub, blk + 1)
          }
          if (blk < nblocks) { TRACK_WS_COMPUTE(ga, ua, blk) }
          #undef TRACK_WS_FETCH
          #undef TRACK_WS_COMPUTE
          for (int row = 0; row < rows; row++) {
            result0[row] = simd_sum(result0[row]);
            result1[row] = simd_sum(result1[row]);
          }
        }

        // Decode down: qmv_reg main loop (K=640, 256-wide blocks) plus the
        // original tail. Prefetch is registers only; the two threadgroup
        // arrays in the caller stay the same size.
        template <typename T, int group_size, int bits, bool EXACT_TAIL = false>
        METAL_FUNC void qmv_reg_ws(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[4]) {
          static_assert(bits == 4, "weight staging: 4-bit");
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          constexpr int nwords = values_per_thread / 4;
          constexpr int WSTEP = block_size * bytes_per_pack / pack_factor;
          constexpr int GSTEP = block_size / group_size;
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
          uint16_t wa[results_per_simdgroup * nwords];
          uint16_t wb[results_per_simdgroup * nwords];
          track_ws_fetch_rows<results_per_simdgroup, nwords>(wa, ws, in_vec_size_w);
          int k = 0;
          int cur = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            thread uint16_t* held = cur == 0 ? &wa[0] : &wb[0];
            if (k + block_size < in_vec_size - block_size) {
              thread uint16_t* nxt = cur == 0 ? &wb[0] : &wa[0];
              track_ws_fetch_rows<results_per_simdgroup, nwords>(
                  nxt, ws + WSTEP, in_vec_size_w);
            }
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot4_from_words<U, nwords>(
                  held + row * nwords, x_thread, s, b, sum);
            }
            ws += WSTEP;
            scales += GSTEP;
            biases += GSTEP;
            x += block_size;
            cur = 1 - cur;
          }
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
              U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device T* sl = scales + row * in_vec_size_g;
                const device T* bl = biases + row * in_vec_size_g;
                U s = sl[0];
                U b = bl[0];
                result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
              }
            }
          } else if (remaining > 0) {
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
        """#

    static let gateUpReuseSource = """
        static_assert(RPS == 1 || RPS == 2, "gate/up row ownership");
        const uint z = threadgroup_position_in_grid.z;
        const bool shared = z == (uint)BR;
        const uint e = shared ? 0u : idx[z];
        const uint r = shared ? 0u : xrow[z];
        const size_t kw = (size_t)KD / 8;
        const size_t kg = (size_t)KD / GS;
        const size_t pitch = (size_t)N * (size_t)TILES;
        const size_t eoff_g = (size_t)e * pitch;
        const size_t eoff_u = TILES > 1 ? eoff_g + (size_t)N : (size_t)e * (size_t)N;
        const device uint32_t* gw = shared ? wsh : wg + eoff_g * kw;
        const device T* gs = shared ? ssh : sg + eoff_g * kg;
        const device T* gb = shared ? bsh : bg + eoff_g * kg;
        const device uint32_t* uw = shared ? wsh + (size_t)N * kw : wu + eoff_u * kw;
        const device T* us = shared ? ssh + (size_t)N * kg : su + eoff_u * kg;
        const device T* ub = shared ? bsh + (size_t)N * kg : bu + eoff_u * kg;
        const int out_row = (int)threadgroup_position_in_grid.y * (2 * RPS)
            + (int)simdgroup_index_in_threadgroup * RPS;
        float g[RPS], u[RPS];
        qmv_fast_reg_dual<T, GS, BITS, RPS>(
            gw, gs, gb, uw, us, ub, x + (size_t)r * (size_t)KD,
            KD, out_row, thread_index_in_simdgroup, g, u);
        if (thread_index_in_simdgroup == 0) {
            for (int i = 0; i < RPS; ++i) {
                const T gv = static_cast<T>(g[i]);
                const T uv = static_cast<T>(u[i]);
                act[(size_t)z * (size_t)N + (size_t)(out_row + i)] = mlx_silu(gv) * uv;
            }
        }
        """

    nonisolated(unsafe) static let gateUpReuseKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up_reuse_2row",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
        outputNames: ["act"],
        source: gateUpReuseSource,
        header: helpersCore + TrackFastKernels.exactHeader + gateUpReuseHelpers,
        ensureRowContiguous: true)

    static let gateUpReuseStagedSource = """
        static_assert(RPS == 2, "weight staging gated to 2 rows/simdgroup");
        const uint z = threadgroup_position_in_grid.z;
        const bool shared = z == (uint)BR;
        const uint e = shared ? 0u : idx[z];
        const uint r = shared ? 0u : xrow[z];
        const size_t kw = (size_t)KD / 8;
        const size_t kg = (size_t)KD / GS;
        const size_t eoff = (size_t)e * (size_t)N;
        const device uint32_t* gw = shared ? wsh : wg + eoff * kw;
        const device T* gs = shared ? ssh : sg + eoff * kg;
        const device T* gb = shared ? bsh : bg + eoff * kg;
        const device uint32_t* uw = shared ? wsh + (size_t)N * kw : wu + eoff * kw;
        const device T* us = shared ? ssh + (size_t)N * kg : su + eoff * kg;
        const device T* ub = shared ? bsh + (size_t)N * kg : bu + eoff * kg;
        const int out_row = (int)threadgroup_position_in_grid.y * (2 * RPS)
            + (int)simdgroup_index_in_threadgroup * RPS;
        float g[RPS], u[RPS];
        qmv_fast_reg_dual_ws<T, GS, BITS, RPS>(
            gw, gs, gb, uw, us, ub, x + (size_t)r * (size_t)KD,
            KD, out_row, thread_index_in_simdgroup, g, u);
        if (thread_index_in_simdgroup == 0) {
            for (int i = 0; i < RPS; ++i) {
                const T gv = static_cast<T>(g[i]);
                const T uv = static_cast<T>(u[i]);
                act[(size_t)z * (size_t)N + (size_t)(out_row + i)] = mlx_silu(gv) * uv;
            }
        }
        """

    nonisolated(unsafe) static let gateUpReuseStagedKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up_reuse_2row_ws",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
        outputNames: ["act"],
        source: gateUpReuseStagedSource,
        header: helpersCore + TrackFastKernels.exactHeader + weightStagingHelpers,
        ensureRowContiguous: true)

    /// Routed slots [0, BR) then the shared expert for the S tokens: act [BR + S, N].
    /// Fused epilogue (`TRACK_SWIGLU_EPILOGUE`, default ON) writes `silu(gate)*up`
    /// from registers. Toggle-off or an unexpected shape uses `gateUp` + `swiglu2`.
    /// `rowsPerSimdgroup` overrides the TRACK_GATEUP_RPS choice (tests); nil uses the toggle.
    /// `staging` overrides TRACK_WEIGHT_STAGING (tests). Nil uses the toggle.
    static func gateUpAct(
        wg: MLXArray, sg: MLXArray, bg: MLXArray, wu: MLXArray, su: MLXArray, bu: MLXArray,
        shared: TrackQuantWeight, x: MLXArray, idx: MLXArray, xrow: MLXArray, groupSize: Int, bits: Int,
        fusedGateUp: Bool = false,
        fusedTiles: Int = 1,
        rowsPerSimdgroup: Int? = nil,
        staging: Bool? = nil
    ) -> MLXArray {
        let tiles = max(fusedTiles > 1 ? fusedTiles : (fusedGateUp ? 2 : 1), 1)
        let N = wg.dim(1) / tiles
        let request = TrackSwiGLUEpilogue.Request(
            dtype: x.dtype, tokens: x.dim(0), hidden: x.dim(1), intermediate: N,
            bits: bits, groupSize: groupSize, sharedRows: shared.rows,
            idxIsUInt32: idx.dtype == .uint32, xrowIsUInt32: xrow.dtype == .uint32)
        // Interleaved gate+up tiles need the fused kernel walk. Split weights
        // keep the SwiGLU-epilogue toggle (fused GEMV vs two launches).
        if tiles > 1 || TrackSwiGLUEpilogue.shouldFuse(request) {
            return gateUpActFused(
                wg: wg, sg: sg, bg: bg, wu: wu, su: su, bu: bu, shared: shared,
                x: x, idx: idx, xrow: xrow, groupSize: groupSize, bits: bits,
                fusedTiles: tiles, rowsPerSimdgroup: rowsPerSimdgroup, staging: staging)
        }
        return gateUpActTwoKernel(
            wg: wg, sg: sg, bg: bg, wu: wu, su: su, bu: bu, shared: shared,
            x: x, idx: idx, xrow: xrow, groupSize: groupSize, bits: bits)
    }

    static func gateUpActFused(
        wg: MLXArray, sg: MLXArray, bg: MLXArray, wu: MLXArray, su: MLXArray, bu: MLXArray,
        shared: TrackQuantWeight, x: MLXArray, idx: MLXArray, xrow: MLXArray, groupSize: Int, bits: Int,
        fusedTiles: Int = 1,
        rowsPerSimdgroup: Int? = nil,
        staging: Bool? = nil
    ) -> MLXArray {
        let BR = idx.dim(0), S = x.dim(0), KD = x.dim(1)
        let tiles = max(fusedTiles, 1)
        let N = wg.dim(1) / tiles
        precondition(N % 8 == 0 && bits == 4 && idx.dtype == .uint32 && xrow.dtype == .uint32)
        precondition(shared.rows == 2 * N && shared.groupSize == groupSize && shared.bits == bits && S >= 1 && S <= 8)
        precondition(isFast(k: KD, n: N), "shared expert one-token path assumes qmv_fast")
        let inputs: [MLXArray] = [
            wg, sg, bg, wu, su, bu, shared.weight, shared.scales, shared.biases!, x, idx, xrow,
        ]
        let route = TrackWeightStaging.gateUpDispatch(
            s: S, dtype: x.dtype, kd: KD, n: N, groupSize: groupSize, bits: bits,
            sharedAffine: shared.mode == .affine,
            enabled: staging ?? TrackWeightStaging.enabled)
        switch route {
        case .reuseStaged, .reuse:
            let rows = rowsPerSimdgroup ?? gateUpReuseRowsPerSimdgroup
            // The staged prefetch kernel is built for the 2-row, split-weight
            // geometry only (static_assert in its source; its expert addressing
            // assumes pitch N, not N * TILES); any other rows or a fused layout
            // keeps the unstaged kernel, whose source supports 1 and 2 rows and
            // the fused pitch, with identical math.
            let staged = route == .reuseStaged
                && rows == TrackWeightStaging.gateUpRowsPerSimdgroup
                && tiles == 1
            let launch = gateUpReuseLaunch(n: N, rowsPerSimdgroup: rows, slots: BR + 1)
            let kernel = staged ? gateUpReuseStagedKernel : gateUpReuseKernel
            return kernel(
                inputs,
                template: [
                    ("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("N", N),
                    ("KD", KD), ("BR", BR), ("RPS", rows), ("TILES", tiles),
                ],
                grid: launch.grid, threadGroup: launch.threadGroup,
                outputShapes: [[BR + S, N]], outputDTypes: [x.dtype])[0]
        case .act:
            return (S == 1 ? gateUpActKernel1 : gateUpActKernel)(
                inputs,
                template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("N", N), ("KD", KD), ("FAST", isFast(k: KD, n: N)), ("BR", BR), ("VPT", S), ("TILES", tiles), ("EXACT_TAIL", useExactTail(k: KD, bits: bits))],
                grid: (32, (N / 8) * 2, BR + 1), threadGroup: (32, 2, 1),
                outputShapes: [[BR + S, N]], outputDTypes: [x.dtype])[0]
        }
    }

    /// Two launches: routed `gateUp` writes both halves, `swiglu2` applies
    /// `mlx_silu(gate)*up`; the shared expert is the same elementwise after its GEMV.
    static func gateUpActTwoKernel(
        wg: MLXArray, sg: MLXArray, bg: MLXArray, wu: MLXArray, su: MLXArray, bu: MLXArray,
        shared: TrackQuantWeight, x: MLXArray, idx: MLXArray, xrow: MLXArray, groupSize: Int, bits: Int
    ) -> MLXArray {
        let n = wg.dim(1)
        let (gate, up) = gateUp(
            wg: wg, sg: sg, bg: bg, wu: wu, su: su, bu: bu,
            x: x, idx: idx, xrow: xrow, groupSize: groupSize, bits: bits)
        let routed = TrackFastKernels.swiglu2(gate: gate, up: up)
        let gu = shared.apply(x)
        let sharedAct = TrackFastKernels.swiglu2(gate: gu[0..., 0..<n], up: gu[0..., n...])
        return concatenated([routed, sharedAct], axis: 0)
    }

    /// Routed experts, down GEMV + expert-weighted combine + shared expert
    /// gate/add (MLX's `col_reduce_small` association over the K experts, as
    /// `track_moe_combine`). act [BR, F], idx uint32 [BR], w f32 [BR] (slot
    /// order), shared [S, H], gate [S] (pre-sigmoid) -> out [S, H].
    /// grid threads (32, H/4, S), threadgroup (32, 1, 1): one simdgroup owns
    /// 4 output columns for one token across all K experts.
    static let downCombineSource = """
        const uint t = threadgroup_position_in_grid.z;
        // MLXFAST-DOWNRPS: RPS output rows per threadgroup (4 for wide windows).
        static_assert(VPT == 1 || RPS == 4, "wide windows keep four rows");
        const int d0 = (int)threadgroup_position_in_grid.y * RPS;
        const uint kw = (uint)F / 8;
        const uint kg = (uint)F / GS;
        const uint sgi = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        // The K expert walks are independent of one another, so they are spread
        // over KSG simdgroups; each product lands in threadgroup memory as the
        // float it already was, and the epilogue folds them in the same k order.
        threadgroup float prod[K][RPS];
        threadgroup float shvT[RPS];
        float res[RPS];
        // K % KSG == 0, so the trip count is the constant K / KSG and the loop
        // still unrolls: each simdgroup keeps that many expert walks in flight.
        for (int kk = 0; kk < K / KSG; ++kk) {
            const int k = (int)sgi + kk * KSG;
            const uint z = t * K + k;
            const uint e = idx[z];
            const size_t eoff = TILES > 2
                ? (size_t)e * (size_t)H * 3 + (size_t)H * 2
                : (size_t)e * (size_t)H;
            const device T* xb = act + (size_t)z * (size_t)F;
            if (FAST) { qmv_fast_reg<T, GS, BITS, RPS>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, lid, res); }
            else { qmv_reg<T, GS, BITS, EXACT_TAIL, RPS>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, lid, res); }
            const float wk = w[z];
            if (lid == 0) {
                for (int i = 0; i < RPS; ++i) { prod[k][i] = static_cast<float>(static_cast<T>(res[i])) * wk; }
            }
        }
        // Shared expert down rows d0..d0+3 for token t: one token routes to `qmv`'s
        // normal branch (K = 640), two to eight to `qmv_wide` (full tiles; a row's
        // walk does not depend on how many vectors share its tile).
        if (sgi == (KSG > K ? (uint)K : 0u)) {
            const device T* xs = act + (size_t)(BR + t) * (size_t)F;
            if constexpr (VPT == 1) {
                float rs[RPS];
                qmv_reg<T, GS, BITS, EXACT_TAIL, RPS>(wsd, ssd, bsd, xs, F, d0, lid, rs);
                if (lid == 0) { for (int i = 0; i < RPS; ++i) { shvT[i] = static_cast<float>(static_cast<T>(rs[i])); } }
            } else {
                float rw[1];
                qmv_wide_reg_full<T, GS, BITS, 1, 8, false>(wsd, ssd, bsd, xs, F, 1, d0 + (int)(lid / 8), lid, rw);
                // The shuffles must run with the whole simdgroup active.
                float sh4[4];
                for (int i = 0; i < 4; ++i) { sh4[i] = static_cast<float>(static_cast<T>(simd_shuffle(rw[0], (ushort)(i * 8)))); }
                if (lid == 0) { for (int i = 0; i < 4; ++i) { shvT[i] = sh4[i]; } }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi == 0 && lid == 0) {
            const T sg = mlx_sigmoid(gate[t]);
            for (int i = 0; i < RPS; ++i) {
                float col[K];
                for (int k = 0; k < K; ++k) { col[k] = prod[k][i]; }
                const T r = static_cast<T>(mlx_colsum_small_f32<K>(col));
                const T sh = sg * static_cast<T>(shvT[i]);
                const T g = r + sh;
                if (HAS_RESIDUAL) {
                    const int d = d0 + i;
                    for (int hc = 0; hc < HC; ++hc) {
                        const T sp = g * inj[(size_t)t * (size_t)HC + (size_t)hc];
                        const T rv = residual[(size_t)t * (size_t)(HC * H)
                            + (size_t)hc * (size_t)H + (size_t)d];
                        out[(size_t)t * (size_t)(HC * H)
                            + (size_t)hc * (size_t)H + (size_t)d] = rv + sp;
                    }
                } else {
                    out[(size_t)t * (size_t)H + (size_t)(d0 + i)] = g;
                }
            }
        }
        """

    nonisolated(unsafe) static let downCombineKernel = MLXFast.metalKernel(
        name: "track_moe_down_combine",
        inputNames: ["wd", "sd", "bd", "wsd", "ssd", "bsd", "act", "idx", "w", "gate", "residual", "inj"],
        outputNames: ["out"],
        source: downCombineSource, header: helpersCore + TrackFastKernels.exactHeader + regHelpers + wideHelpers,
        ensureRowContiguous: true)
    nonisolated(unsafe) static let downCombineKernel1 = MLXFast.metalKernel(
        name: "track_moe_down_combine_1",
        inputNames: ["wd", "sd", "bd", "wsd", "ssd", "bsd", "act", "idx", "w", "gate", "residual", "inj"],
        outputNames: ["out"],
        source: downCombineSource, header: helpersCore + TrackFastKernels.exactHeader + regHelpers + wideDecls,
        ensureRowContiguous: true)

    /// Same epilogue as `downCombineSource`. Routed and shared `qmv_reg` walks
    /// prefetch the next K-block's uint16 words. Threadgroup arrays stay
    /// `prod[K][4]` and `shvT[4]`. FAST stays the original helper (dead at F=640).
    static let downCombineStagedSource = """
        const uint t = threadgroup_position_in_grid.z;
        const int d0 = (int)threadgroup_position_in_grid.y * 4;
        const uint kw = (uint)F / 8;
        const uint kg = (uint)F / GS;
        const uint sgi = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup float prod[K][4];
        threadgroup float shvT[4];
        float res[4];
        for (int kk = 0; kk < K / KSG; ++kk) {
            const int k = (int)sgi + kk * KSG;
            const uint z = t * K + k;
            const uint e = idx[z];
            const size_t eoff = (size_t)e * (size_t)H;
            const device T* xb = act + (size_t)z * (size_t)F;
            if (FAST) { qmv_fast_reg<T, GS, BITS>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, lid, res); }
            else { qmv_reg_ws<T, GS, BITS, (F % get_pack_factor<BITS, 32>()) == 0>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, lid, res); }
            const float wk = w[z];
            if (lid == 0) {
                for (int i = 0; i < 4; ++i) { prod[k][i] = static_cast<float>(static_cast<T>(res[i])) * wk; }
            }
        }
        if (sgi == (KSG > K ? (uint)K : 0u)) {
            const device T* xs = act + (size_t)(BR + t) * (size_t)F;
            if constexpr (VPT == 1) {
                float rs[4];
                qmv_reg_ws<T, GS, BITS, (F % get_pack_factor<BITS, 32>()) == 0>(wsd, ssd, bsd, xs, F, d0, lid, rs);
                if (lid == 0) { for (int i = 0; i < 4; ++i) { shvT[i] = static_cast<float>(static_cast<T>(rs[i])); } }
            } else {
                float rw[1];
                qmv_wide_reg_full<T, GS, BITS, 1, 8, false>(wsd, ssd, bsd, xs, F, 1, d0 + (int)(lid / 8), lid, rw);
                float sh4[4];
                for (int i = 0; i < 4; ++i) { sh4[i] = static_cast<float>(static_cast<T>(simd_shuffle(rw[0], (ushort)(i * 8)))); }
                if (lid == 0) { for (int i = 0; i < 4; ++i) { shvT[i] = sh4[i]; } }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgi == 0 && lid == 0) {
            const T sg = mlx_sigmoid(gate[t]);
            for (int i = 0; i < 4; ++i) {
                float col[K];
                for (int k = 0; k < K; ++k) { col[k] = prod[k][i]; }
                const T r = static_cast<T>(mlx_colsum_small_f32<K>(col));
                const T sh = sg * static_cast<T>(shvT[i]);
                out[(size_t)t * (size_t)H + (size_t)(d0 + i)] = r + sh;
            }
        }
        """

    nonisolated(unsafe) static let downCombineStagedKernel1 = MLXFast.metalKernel(
        name: "track_moe_down_combine_1_ws",
        inputNames: ["wd", "sd", "bd", "wsd", "ssd", "bsd", "act", "idx", "w", "gate"],
        outputNames: ["out"],
        source: downCombineStagedSource,
        header: helpersCore + TrackFastKernels.exactHeader + regHelpers + weightStagingHelpers + wideDecls,
        ensureRowContiguous: true)

    /// Simdgroups per down+combine threadgroup: the top-K expert walks of one
    /// output tile are handed out round-robin over this many simdgroups, which
    /// shortens each simdgroup's serial chain and drops the per-thread product
    /// array into threadgroup memory. Every product and the fold order are
    /// unchanged, so the output is bit-identical for any value.
    /// MLXFAST-DOWNRPS: output rows per down+combine threadgroup in a one-token
    /// window. Each row's expert walks and the fold are unchanged for any value;
    /// fewer rows per threadgroup means more threadgroups in flight (H / rows).
    static let downRowsPerSimdgroup = 4

    static let downCombineSimdgroups =
        ProcessInfo.processInfo.environment["MLXFAST_MOE_DOWN_SIMDGROUPS"].flatMap { Int($0) } ?? 5

    /// act [BR + S, F] (routed slots, then the shared expert per token), gate [S] pre-sigmoid.
    /// When `residual` and `inject` match the decode residual-epilogue gate the
    /// kernel writes the W-wide stream (`residual + (down) * inject`) instead
    /// of the H-wide block output.
    /// `staging` overrides TRACK_WEIGHT_STAGING (tests). Nil uses the toggle.
    static func downCombine(
        wd: MLXArray, sd: MLXArray, bd: MLXArray, sharedDown: TrackQuantWeight, act: MLXArray,
        idx: MLXArray, w: MLXArray, gate: MLXArray, topK: Int, groupSize: Int, bits: Int,
        residual: MLXArray? = nil, inject: MLXArray? = nil, hcCount: Int = 4,
        staging: Bool? = nil,
        fusedTiles: Int = 1, rowsPerDown: Int = 0
    ) -> MLXArray {
        let BR = idx.dim(0), F = act.dim(1)
        var wdUse = wd, sdUse = sd, bdUse = bd
        if fusedTiles > 2, rowsPerDown > 0 {
            let expertCount = wd.dim(0)
            let rowsPerHalf = wd.dim(1) / 3
            let packed = wd.dim(2)
            let tile = rowsPerHalf * packed
            let downInner = tile / rowsPerDown
            wdUse = wd.reshaped(expertCount, 3 * rowsPerDown, downInner)
            let scaleInner = sd.dim(2)
            let downGroups = (rowsPerHalf * scaleInner) / rowsPerDown
            sdUse = sd.reshaped(expertCount, 3 * rowsPerDown, downGroups)
            bdUse = bd.reshaped(expertCount, 3 * rowsPerDown, downGroups)
        }
        let H = fusedTiles > 2 ? rowsPerDown : wdUse.dim(1)
        let S = BR / topK
        precondition(BR % topK == 0 && H % 4 == 0 && bits == 4 && w.dtype == .float32 && S >= 1 && S <= 8)
        precondition(act.dim(0) == BR + S && gate.dim(0) == S && sharedDown.rows == H && !isFast(k: F, n: H))
        let fuse = residual != nil && inject != nil
            && TrackResidualEpilogue.coversDownProj(k: F, n: H, tokens: S)
            && residual!.dim(-1) == hcCount * H && inject!.dim(-1) == hcCount
        let ksg = topK % downCombineSimdgroups == 0 ? downCombineSimdgroups : 1
        let rps = S == 1 ? downRowsPerSimdgroup : 4
        let dummy = act
        let tiles = max(fusedTiles, 1)
        // The staged down kernel bakes its build geometry (4 output rows per
        // threadgroup); if the dispatch rows ever differ, fall back to the
        // parameterized unstaged kernel instead of misinstantiating.
        let staged = !fuse && fusedTiles <= 2 && rps == TrackWeightStaging.downRowsPerThreadgroup
            && TrackWeightStaging.downDispatch(
                windowTokens: S, f: F, h: H, bits: bits,
                enabled: staging ?? TrackWeightStaging.enabled) == .combineStaged
        let kernel = staged ? downCombineStagedKernel1 : (S == 1 ? downCombineKernel1 : downCombineKernel)
        let inputs: [MLXArray] =
            staged
            ? [wdUse, sdUse, bdUse, sharedDown.weight, sharedDown.scales, sharedDown.biases!, act, idx, w, gate]
            : [wdUse, sdUse, bdUse, sharedDown.weight, sharedDown.scales, sharedDown.biases!, act, idx, w, gate,
               residual ?? dummy, inject ?? dummy]
        return kernel(
            inputs,
            template: [
                ("T", act.dtype), ("GS", groupSize), ("BITS", bits), ("H", H), ("F", F), ("K", topK),
                ("FAST", isFast(k: F, n: H)), ("BR", BR), ("VPT", S), ("KSG", ksg), ("RPS", rps),
                ("HAS_RESIDUAL", fuse), ("HC", hcCount),
                ("EXACT_TAIL", useExactTail(k: F, bits: bits)), ("TILES", tiles),
            ],
            grid: (32, (H / rps) * ksg, S), threadGroup: (32, ksg, 1),
            outputShapes: [fuse ? [S, hcCount * H] : [S, H]], outputDTypes: [act.dtype])[0]
    }
}

// MARK: qmv_wide replica (2-8 token windows)
extension TrackFastMoEKernels {
    static let wideHelpers = #"""
        // MLX `qmv_wide_impl` (the M >= 2 quantized GEMV on this GPU generation),
        // verbatim for a FULL 8-row tile (k_folds == 1): same lane -> group
        // assignment (k_lane, stride k_lanes), same per-group decode and
        // element-order accumulation, same shuffle ladder. The row is given by
        // the caller; the vecs_per_tg vectors are x's first rows. The totals
        // are left in `result` on the k_lane == 0 lanes.
        template <typename T> METAL_FUNC vec<T, 4> track_silu4(vec<T, 4> x) {
            return vec<T, 4>(mlx_silu(x.x), mlx_silu(x.y), mlx_silu(x.z), mlx_silu(x.w));
        }
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, bool SILU, int k_bound = 0>
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
          // k_bound > 0: compile-time K from the mixer staging generator.
          // k_bound == 0: keep the runtime in_vec_size argument.
          const int K = k_bound > 0 ? k_bound : in_vec_size;
        
          const int in_vec_size_w = K * bits / 8; // bytes per weight row
          const int in_vec_size_g = K / group_size;
          const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
          const device T* srow = scales + row * in_vec_size_g;
          const device T* brow = biases + row * in_vec_size_g;
        
          const device T* xv[vecs_per_tg];
          for (int v = 0; v < vecs_per_tg; v++) {
            xv[v] = x + min(vec0 + v, M - 1) * K;
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
          } else {
            for (int g = g_first; g < in_vec_size_g; g += g_stride) {
              U scale = srow[g];
              U bias = brow[g];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const device uint8_t* wc = wrow + k0 * bits / 8;
                U w_dq[sub];
                dequantize<U, sub, bits>(wc, scale, bias, w_dq);
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
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, int k_bound = 0, int n_bound = 0>
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
          const int K = k_bound > 0 ? k_bound : in_vec_size;
          const int N = n_bound > 0 ? n_bound : out_vec_size;
          const int tile_rows = min(N - tile_row0, rows_per_tg);
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
          const int row = min(out_row, N - 1);
        
          const int in_vec_size_w = K * bits / 8; // bytes per weight row
          const int in_vec_size_g = K / group_size;
          const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
          const device T* srow = scales + row * in_vec_size_g;
          const device T* brow = biases + row * in_vec_size_g;
        
          const device T* xv[vecs_per_tg];
          for (int v = 0; v < vecs_per_tg; v++) {
            xv[v] = x + min(vec0 + v, M - 1) * K;
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
          } else {
            for (int g = g_first; g < in_vec_size_g; g += g_stride) {
              U scale = srow[g];
              U bias = brow[g];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const device uint8_t* wc = wrow + k0 * bits / 8;
                U w_dq[sub];
                dequantize<U, sub, bits>(wc, scale, bias, w_dq);
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
          if (k_lane == 0 && fold == 0 && out_row < N) { valid = true; }
        }
        """#

    /// Declarations only: a one-token kernel instantiation never reaches its
    /// wide (2-8 token) branch, so it needs the names, not the 27 KB of bodies
    /// (the custom-kernel call cost on the host scales with the source size).
    static let wideDecls = #"""
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, bool SILU, int k_bound = 0>
        METAL_FUNC void qmv_wide_reg_full(
            const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
            const int in_vec_size, const int M, const int row, uint simd_lid, thread float (&result)[vecs_per_tg]);
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, int k_bound = 0, int n_bound = 0>
        METAL_FUNC void qmv_wide_reg_partial(
            const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
            const int in_vec_size, const int out_vec_size, const int M, threadgroup float* fold_partials,
            uint simd_gid, uint simd_lid, thread float (&result)[vecs_per_tg], thread bool& valid, thread int& row_out);
        """#

    /// Check kernel: y [M, N] = x [M, K] * W^T over `qmv_wide_reg_full`, tiles of
    /// 8 rows (2 simdgroups x 4 slots), to test the replica against MLX.
    static let wideCheckSource = """
        const int tile_row0 = (int)threadgroup_position_in_grid.y * 8;
        const int slot = (int)simdgroup_index_in_threadgroup * 4 + (int)(thread_index_in_simdgroup / 8);
        const int row = tile_row0 + slot;
        float result[VPT];
        qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(w, scales, biases, x, K, M, row, thread_index_in_simdgroup, result);
        if ((thread_index_in_simdgroup % 8) == 0) {
            for (int v = 0; v < VPT; v++) {
                if (v < M) { y[(size_t)v * (size_t)N + (size_t)row] = static_cast<T>(result[v]); }
            }
        }
        """

    nonisolated(unsafe) static let wideCheckKernel = MLXFast.metalKernel(
        name: "track_qmv_wide_check",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: wideCheckSource, header: helpers + TrackFastKernels.exactHeader + wideHelpers, ensureRowContiguous: true)

    static func wideCheck(w: MLXArray, scales: MLXArray, biases: MLXArray, x: MLXArray, groupSize: Int, bits: Int) -> MLXArray {
        let M = x.dim(0), K = x.dim(1), N = w.dim(0)
        precondition(N % 8 == 0 && M >= 1 && M <= 8 && bits == 4)
        return wideCheckKernel(
            [w, scales, biases, x],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("VPT", M), ("K", K), ("N", N), ("M", M)],
            grid: (32, (N / 8) * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[M, N]], outputDTypes: [x.dtype])[0]
    }
}

extension TrackFastMoEKernels {
    /// Check kernel for the short-tile replica: y [M, N] with N < 8.
    static let widePartialCheckSource = """
        threadgroup float fold_partials[8 * VPT];
        float result[VPT];
        bool valid = false; int row = 0;
        qmv_wide_reg_partial<T, GS, BITS, VPT, 8>(w, scales, biases, x, K, N, M, fold_partials, simdgroup_index_in_threadgroup, thread_index_in_simdgroup, result, valid, row);
        if (valid) {
            for (int v = 0; v < VPT; v++) {
                if (v < M) { y[(size_t)v * (size_t)N + (size_t)row] = static_cast<T>(result[v]); }
            }
        }
        """

    nonisolated(unsafe) static let widePartialCheckKernel = MLXFast.metalKernel(
        name: "track_qmv_wide_partial_check",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: widePartialCheckSource, header: helpers + TrackFastKernels.exactHeader + wideHelpers, ensureRowContiguous: true)

    static func widePartialCheck(w: MLXArray, scales: MLXArray, biases: MLXArray, x: MLXArray, groupSize: Int, bits: Int) -> MLXArray {
        let M = x.dim(0), K = x.dim(1), N = w.dim(0)
        precondition(N < 8 && M >= 1 && M <= 8 && bits == 4)
        return widePartialCheckKernel(
            [w, scales, biases, x],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("VPT", M), ("K", K), ("N", N), ("M", M)],
            grid: (32, 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[M, N]], outputDTypes: [x.dtype])[0]
    }
}

// MARK: exact-tail check kernels (qmv_impl + inject small-N vs the vendored path)
extension TrackFastMoEKernels {
    /// y [M, N] = x [M, K] * W^T over the Runner `qmv_impl` copy. `EXACT_TAIL`
    /// selects the unchecked tail; false is the vendored `_safe` tail.
    static let qmvImplCheckSource = """
        const uint b = threadgroup_position_in_grid.z;
        const uint blk = threadgroup_position_in_grid.y;
        uint3 tid = uint3(0, blk, 0);
        const device T* xb = x + (size_t)b * (size_t)K;
        device T* yb = y + (size_t)b * (size_t)N;
        qmv_impl<T, GS, BITS, EXACT_TAIL>(
            w, scales, biases, xb, yb, K, N, tid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """

    nonisolated(unsafe) static let qmvImplCheckKernel = MLXFast.metalKernel(
        name: "track_qmv_impl_check",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: qmvImplCheckSource, header: helpers, ensureRowContiguous: true)

    static func qmvImplCheck(
        w: MLXArray, scales: MLXArray, biases: MLXArray, x: MLXArray,
        groupSize: Int, bits: Int, exactTail: Bool
    ) -> MLXArray {
        let M = x.dim(0), K = x.dim(1), N = w.dim(0)
        precondition(N % 8 == 0 && M >= 1 && bits == 4)
        let nb = N / 8
        return qmvImplCheckKernel(
            [w, scales, biases, x],
            template: [
                ("T", x.dtype), ("GS", groupSize), ("BITS", bits),
                ("K", K), ("N", N), ("EXACT_TAIL", exactTail),
            ],
            grid: (32, nb * 2, M), threadGroup: (32, 2, 1),
            outputShapes: [[M, N]], outputDTypes: [x.dtype])[0]
    }

    /// y [M, N] = x [M, K] * W^T over `track_inject_qmv` (small-N branch, N < 8).
    static let injectCheckSource = """
        const uint row = threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        track_inject_qmv<T, GS, BITS, K, N, 4, EXACT_TAIL>(
            w, scales, biases, x + (size_t)row * (size_t)K,
            y + (size_t)row * (size_t)N, sg, lid);
        """

    nonisolated(unsafe) static let injectCheckKernel = MLXFast.metalKernel(
        name: "track_inject_qmv_check",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: injectCheckSource,
        header: TrackFastKernels.mixerHeadHeader,
        ensureRowContiguous: true)

    static func injectCheck(
        w: MLXArray, scales: MLXArray, biases: MLXArray, x: MLXArray,
        groupSize: Int, bits: Int, exactTail: Bool
    ) -> MLXArray {
        let M = x.dim(0), K = x.dim(1), N = w.dim(0)
        precondition(N < 8 && M >= 1 && bits == 4 && K > 256 && K % 256 == 0)
        return injectCheckKernel(
            [w, scales, biases, x],
            template: [
                ("T", x.dtype), ("GS", groupSize), ("BITS", bits),
                ("K", K), ("N", N), ("EXACT_TAIL", exactTail),
            ],
            grid: (64, M, 1), threadGroup: (64, 1, 1),
            outputShapes: [[M, N]], outputDTypes: [x.dtype])[0]
    }
}

// MARK: router GEMV for one-token windows: MLX's float `gemv` kernel
//       (GEMVKernel<float, BM=4, BN=1, SM=1, SN=32, TM=4, TN=4>, the
//       parameters `gemv_axbpy` selects for a [512 x 2560] matrix and a
//       2560-vector). Each row keeps that walk while the target shape assigns
//       two rows per SIMD group. Reading bf16 weights instead of a float32
//       copy preserves every product and partial sum.

extension TrackFastMoEKernels {
    static let routerGemvSource = """
        constexpr int TM = RPS, TN = 4, SN = 32, blockM = 4 * RPS, blockN = 128;
        const int tid_x = (int)threadgroup_position_in_grid.x;
        const int simd_gid = (int)simdgroup_index_in_threadgroup;
        const int simd_lid = (int)thread_index_in_simdgroup;
        float result[TM] = {0};
        float inter[TN];
        float v_coeff[TN];
        const int thrN = simd_lid;           // SN == 32: thrM = 0
        const int simdM = simd_gid;          // SM == 1, BN == 1
        int bm = simdM * TM;
        int bn = thrN * TN;
        int out_row = tid_x * blockM + bm;
        if (out_row >= N) return;
        out_row = out_row + TM <= N ? out_row : N - TM;
        const device T* mat = w + (size_t)out_row * (size_t)K;
        const int n_iter = K / blockN;
        for (int i = 0; i < n_iter; ++i) {
            for (int tn = 0; tn < TN; tn++) { v_coeff[tn] = x[bn + tn]; }
            int mat_offset = 0;
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) { inter[tn] = static_cast<float>(mat[mat_offset + bn + tn]); }
                for (int tn = 0; tn < TN; tn++) { result[tm] += inter[tn] * v_coeff[tn]; }
                mat_offset += K;
            }
            bn += blockN;
        }
        for (int tm = 0; tm < TM; tm++) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        if (simd_lid == 0) {
            for (int tm = 0; tm < TM; tm++) { out[out_row + tm] = result[tm]; }
        }
        """

    nonisolated(unsafe) static let routerGemvKernel = MLXFast.metalKernel(
        name: "track_router_gemv",
        inputNames: ["x", "w"],
        outputNames: ["out"],
        source: routerGemvSource, ensureRowContiguous: true)

    /// x float32 [K], w bf16 [N, K] -> logits float32 [N]. One-token windows only
    /// Retains MLX's per-row arithmetic for K in [65, 16N) with N < 4096.
    static func routerGemv(x: MLXArray, w: MLXArray) -> MLXArray {
        let K = w.dim(1), N = w.dim(0)
        precondition(x.dtype == .float32 && x.size == K && w.dtype == .bfloat16)
        precondition(K % 128 == 0 && K > 64 && K < 16 * N && N % 16 == 0 && N < 4096)
        let rowsPerSimdgroup = K == 2560 && N == 512 ? 1 : 4  // MLXFAST-ROUTERRPS1
        return routerGemvKernel(
            [x.reshaped(K), w],
            template: [("T", w.dtype), ("K", K), ("N", N), ("RPS", rowsPerSimdgroup)],
            grid: (32 * (N / (4 * rowsPerSimdgroup)), 1, 4), threadGroup: (32, 1, 4),
            outputShapes: [[N]], outputDTypes: [.float32])[0]
    }
}

// MARK: software-pipelined qmv_fast (experiment): same lanes, same accumulation,
//       the next block's packs/scales/biases/x loaded before this block's qdots.

extension TrackFastMoEKernels {
    static let pipelinedHelpers = #"""

        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_fast_reg_pf(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
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
          static_assert(bits == 4 && packs_per_thread == 2, "pipelined path: 4-bit");
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
          // Prefetched raw operands of the NEXT block: the two 32-bit packs, the
          // scale and the bias of each of the 4 rows, and the 16 activations.
          uint32_t pk[4][2];
          T sc[4], bi[4];
          T xr[values_per_thread];
          auto fetch = [&](const device uint8_t* wsb, const device T* scb, const device T* bib, const device T* xb) {
            for (int row = 0; row < 4; row++) {
              const device uint32_t* wl = (const device uint32_t*)(wsb + row * in_vec_size_w);
              pk[row][0] = wl[0]; pk[row][1] = wl[1];
              sc[row] = scb[row * in_vec_size_g];
              bi[row] = bib[row * in_vec_size_g];
            }
            for (int i = 0; i < values_per_thread; i++) { xr[i] = xb[i]; }
          };
          fetch(ws, scales, biases, x);
          for (int k = 0; k < in_vec_size; k += block_size) {
            // hold this block's operands, then issue the next block's loads
            uint32_t cpk[4][2]; T csc[4], cbi[4]; T cx[values_per_thread];
            for (int row = 0; row < 4; row++) { cpk[row][0] = pk[row][0]; cpk[row][1] = pk[row][1]; csc[row] = sc[row]; cbi[row] = bi[row]; }
            for (int i = 0; i < values_per_thread; i++) { cx[i] = xr[i]; }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
            if (k + block_size < in_vec_size) { fetch(ws, scales, biases, x); }
            // load_vector on the held activations (same expression as load_vector)
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += cx[i] + cx[i + 1] + cx[i + 2] + cx[i + 3];
              x_thread[i] = cx[i];
              x_thread[i + 1] = cx[i + 1] / 16.0f;
              x_thread[i + 2] = cx[i + 2] / 256.0f;
              x_thread[i + 3] = cx[i + 3] / 4096.0f;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = csc[row];
              U b = cbi[row];
              // qdot over the held packs: same expression as qdot<U, 16, 4>
              U accum = 0;
              const thread uint16_t* wsh = (const thread uint16_t*)&cpk[row][0];
              for (int i = 0; i < (values_per_thread / 4); i++) {
                accum +=
                    (x_thread[4 * i] * (wsh[i] & 0x000f) +
                     x_thread[4 * i + 1] * (wsh[i] & 0x00f0) +
                     x_thread[4 * i + 2] * (wsh[i] & 0x0f00) +
                     x_thread[4 * i + 3] * (wsh[i] & 0xf000));
              }
              result[row] += s * accum + sum * b;
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        // Ping-pong variant: two register sets, the K loop unrolled by two, no
        // per-block copies. Same arithmetic as qmv_fast (see qmv_fast_reg_pf).
        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_fast_reg_pf2(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int packs_per_thread = 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(bits == 4, "4-bit");
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          for (int row = 0; row < 4; row++) { result[row] = 0; }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          constexpr int WSTEP = block_size * bytes_per_pack / pack_factor;
          constexpr int GSTEP = block_size / group_size;
          uint32_t pa[4][2], pb[4][2];
          T sa[4], ba[4], sb[4], bb[4];
          T xa[values_per_thread], xb[values_per_thread];
          #define TRACK_FETCH(PK, SC, BI, XR, OFFB) \
            for (int row = 0; row < 4; row++) { \
              const device uint32_t* wl = (const device uint32_t*)(ws + (OFFB) * WSTEP + row * in_vec_size_w); \
              PK[row][0] = wl[0]; PK[row][1] = wl[1]; \
              SC[row] = scales[(OFFB) * GSTEP + row * in_vec_size_g]; \
              BI[row] = biases[(OFFB) * GSTEP + row * in_vec_size_g]; \
            } \
            for (int i = 0; i < values_per_thread; i++) { XR[i] = x[(OFFB) * block_size + i]; }
          #define TRACK_COMPUTE(PK, SC, BI, XR) { \
            U x_thread[values_per_thread]; \
            U sum = 0; \
            for (int i = 0; i < values_per_thread; i += 4) { \
              sum += XR[i] + XR[i + 1] + XR[i + 2] + XR[i + 3]; \
              x_thread[i] = XR[i]; \
              x_thread[i + 1] = XR[i + 1] / 16.0f; \
              x_thread[i + 2] = XR[i + 2] / 256.0f; \
              x_thread[i + 3] = XR[i + 3] / 4096.0f; \
            } \
            for (int row = 0; row < 4; row++) { \
              U s = SC[row]; U b = BI[row]; U accum = 0; \
              const thread uint16_t* wsh = (const thread uint16_t*)&PK[row][0]; \
              for (int i = 0; i < (values_per_thread / 4); i++) { \
                accum += (x_thread[4 * i] * (wsh[i] & 0x000f) + x_thread[4 * i + 1] * (wsh[i] & 0x00f0) + \
                          x_thread[4 * i + 2] * (wsh[i] & 0x0f00) + x_thread[4 * i + 3] * (wsh[i] & 0xf000)); \
              } \
              result[row] += s * accum + sum * b; \
            } }
          const int nblocks = in_vec_size / block_size;
          TRACK_FETCH(pa, sa, ba, xa, 0)
          int blk = 0;
          for (; blk + 1 < nblocks; blk += 2) {
            TRACK_FETCH(pb, sb, bb, xb, blk + 1)
            TRACK_COMPUTE(pa, sa, ba, xa)
            if (blk + 2 < nblocks) { TRACK_FETCH(pa, sa, ba, xa, blk + 2) }
            TRACK_COMPUTE(pb, sb, bb, xb)
          }
          if (blk < nblocks) { TRACK_COMPUTE(pa, sa, ba, xa) }
          #undef TRACK_FETCH
          #undef TRACK_COMPUTE
          for (int row = 0; row < 4; row++) { result[row] = simd_sum(result[row]); }
        }
        """#

}
