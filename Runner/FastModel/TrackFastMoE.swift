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
    static let helpers = #"""
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

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
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

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
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
    if (remaining > 0) {
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
    if (remaining > 0) {
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
            else { qmv_impl<T, GS, BITS>(wb, sb, bb, xb, yb_, K, N, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup); }
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
            else { qmv_impl<T, GS, BITS>(wb, sb, bb, xb, yb_, K, N, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup); }
        }
        """

    nonisolated(unsafe) static let gateUpKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "x", "idx", "xrow", "K", "N"],
        outputNames: ["gate", "up"],
        source: gateUpSource, header: helpers, ensureRowContiguous: true)

    nonisolated(unsafe) static let singleKernel = MLXFast.metalKernel(
        name: "track_moe_single",
        inputNames: ["w", "scales", "biases", "x", "idx", "K", "N"],
        outputNames: ["out"],
        source: singleSource, header: helpers, ensureRowContiguous: true)

    static func isFast(k: Int, n: Int) -> Bool { n % 8 == 0 && k % 512 == 0 }

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
            [wg, sg, bg, wu, su, bu, x, idx, xrow, MLXArray(Int32(K)), MLXArray(Int32(N))],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("FAST", isFast(k: K, n: N)), ("RB", rb)],
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
            [w, scales, biases, x, idx, MLXArray(Int32(K)), MLXArray(Int32(N))],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("FAST", isFast(k: K, n: N)), ("RB", rb)],
            grid: (32, nb * 2, B), threadGroup: (32, 2, 1),
            outputShapes: [[B, N]], outputDTypes: [x.dtype])[0]
    }
}

// MARK: router tail: top-k selection + softmax in ONE launch
//
// Replaces `argPartition(-logits, kth: k-1)[..., ..<k]`, `takeAlong`,
// `softmax(precise: true)` and the index cast. MLX's GPU argpartition is a
// full stable merge sort (`sort.h`: strict `<` in both the thread sort and
// the merge step), so the first k entries are the k largest logits in
// descending order with ties in ascending index order; the kernel selects
// exactly that. The softmax is `softmax_single_row` verbatim for a row of k
// (N_READS = 4, one simdgroup, `fast::exp`, per-thread sequential partials,
// `simd_sum`, multiply by the reciprocal).

extension TrackFastMoEKernels {
    /// logits f32 [R, E] -> idx uint32 [R, K], w f32 [R, K]. grid threads (32, R, 1), tg (32,1,1).
    static let routeSource = """
        constexpr int E_PER = (E + 31) / 32;
        const uint row = thread_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
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
        threadgroup float selv[K];
        threadgroup uint seli[K];
        for (int k = 0; k < K; ++k) {
            // lane-local best: largest value, then lowest index
            float bv = -INFINITY; int bj = -1;
            for (int j = 0; j < E_PER; ++j) {
                if (!taken[j] && (v[j] > bv)) { bv = v[j]; bj = j; }
            }
            const float gmax = simd_max(bv);
            const uint cand = (bv == gmax && bj >= 0) ? (uint)(lane + 32 * bj) : 0xffffffffu;
            const uint gidx = simd_min(cand);
            if (lane == 0) { selv[k] = gmax; seli[k] = gidx; }
            if (gidx == (uint)(lane + 32 * bj) && bj >= 0) { taken[bj] = true; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
        maxval = simd_max((lane == 0) ? maxval : -INFINITY);
        float normalizer = 0;
        for (int i = 0; i < N_READS; i++) {
            float exp_x = fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        normalizer = simd_sum((lane == 0) ? normalizer : 0.0f);
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
        inputNames: ["logits"],
        outputNames: ["idx", "w"],
        source: routeSource, ensureRowContiguous: true)

    /// logits f32 [..., E] -> (idx uint32 [..., K], w f32 [..., K])
    static func route(logits: MLXArray, topK: Int) -> (idx: MLXArray, w: MLXArray) {
        precondition(logits.dtype == .float32)
        let E = logits.dim(-1)
        let lead = Array(logits.shape.dropLast())
        let R = lead.reduce(1, *)
        precondition(topK <= 32 && topK <= E)
        let outs = routeKernel(
            [logits.reshaped(R, E)],
            template: [("E", E), ("K", topK)],
            grid: (32, R, 1), threadGroup: (32, 1, 1),
            outputShapes: [[R, topK], [R, topK]], outputDTypes: [.uint32, .float32])
        return (outs[0].reshaped(lead + [topK]), outs[1].reshaped(lead + [topK]))
    }
}

// MARK: fused routed-expert MLP: gate|up GEMVs + SwiGLU in one launch, down GEMV
//       + expert-weighted combine in one launch. The GEMV walks are MLX's own
//       `qmv_fast_impl` / `qmv_impl` (normal branch) with the result kept in
//       registers instead of stored: same lanes, same accumulation, same
//       `simd_sum`, same bf16 rounding of each expert's output before the
//       epilogue arithmetic (`static_cast<T>`), which is what the separate
//       launches did.

extension TrackFastMoEKernels {
    static let regHelpers = #"""

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
        """#

    /// Routed experts, gate|up + SwiGLU. x [S, KD], idx/xrow uint32 [BR] ->
    /// act [BR, N]. grid threads (32, N/8, BR), threadgroup (32, 2, 1).
    static let gateUpActSource = """
        const uint z = threadgroup_position_in_grid.z;
        const uint e = idx[z];
        const uint r = xrow[z];
        const uint kw = (uint)KD / 8;
        const uint kg = (uint)KD / GS;
        const int out_row = (int)threadgroup_position_in_grid.y * 8 + (int)simdgroup_index_in_threadgroup * 4;
        const device T* xb = x + (size_t)r * (size_t)KD;
        const size_t eoff = (size_t)e * (size_t)N;
        float g[4], u[4];
        if (FAST) {
            qmv_fast_reg<T, GS, BITS>(wg + eoff * kw, sg + eoff * kg, bg + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, g);
            qmv_fast_reg<T, GS, BITS>(wu + eoff * kw, su + eoff * kg, bu + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, u);
        } else {
            qmv_reg<T, GS, BITS>(wg + eoff * kw, sg + eoff * kg, bg + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, g);
            qmv_reg<T, GS, BITS>(wu + eoff * kw, su + eoff * kg, bu + eoff * kg, xb, KD, out_row, thread_index_in_simdgroup, u);
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
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "x", "idx", "xrow"],
        outputNames: ["act"],
        source: gateUpActSource, header: helpers + TrackFastKernels.exactHeader + regHelpers,
        ensureRowContiguous: true)

    static func gateUpAct(
        wg: MLXArray, sg: MLXArray, bg: MLXArray, wu: MLXArray, su: MLXArray, bu: MLXArray,
        x: MLXArray, idx: MLXArray, xrow: MLXArray, groupSize: Int, bits: Int
    ) -> MLXArray {
        let BR = idx.dim(0), KD = x.dim(1), N = wg.dim(1)
        precondition(N % 8 == 0 && bits == 4 && idx.dtype == .uint32 && xrow.dtype == .uint32)
        return gateUpActKernel(
            [wg, sg, bg, wu, su, bu, x, idx, xrow],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("N", N), ("KD", KD), ("FAST", isFast(k: KD, n: N))],
            grid: (32, (N / 8) * 2, BR), threadGroup: (32, 2, 1),
            outputShapes: [[BR, N]], outputDTypes: [x.dtype])[0]
    }

    /// Routed experts, down GEMV + expert-weighted combine + shared expert
    /// gate/add (MLX's `col_reduce_small` association over the K experts, as
    /// `track_moe_combine`). act [BR, F], idx uint32 [BR], w f32 [BR] (slot
    /// order), shared [S, H], gate [S] (pre-sigmoid) -> out [S, H].
    /// grid threads (32, H/4, S), threadgroup (32, 1, 1): one simdgroup owns
    /// 4 output columns for one token across all K experts.
    static let downCombineSource = """
        const uint t = threadgroup_position_in_grid.z;
        const int d0 = (int)threadgroup_position_in_grid.y * 4;
        const uint kw = (uint)F / 8;
        const uint kg = (uint)F / GS;
        float prod[K][4];
        float res[4];
        for (int k = 0; k < K; ++k) {
            const uint z = t * K + k;
            const uint e = idx[z];
            const size_t eoff = (size_t)e * (size_t)H;
            const device T* xb = act + (size_t)z * (size_t)F;
            if (FAST) { qmv_fast_reg<T, GS, BITS>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, thread_index_in_simdgroup, res); }
            else { qmv_reg<T, GS, BITS>(wd + eoff * kw, sd + eoff * kg, bd + eoff * kg, xb, F, d0, thread_index_in_simdgroup, res); }
            const float wk = w[z];
            for (int i = 0; i < 4; ++i) { prod[k][i] = static_cast<float>(static_cast<T>(res[i])) * wk; }
        }
        if (thread_index_in_simdgroup == 0) {
            const T sg = mlx_sigmoid(gate[t]);
            for (int i = 0; i < 4; ++i) {
                float col[K];
                for (int k = 0; k < K; ++k) { col[k] = prod[k][i]; }
                const T r = static_cast<T>(mlx_colsum_small_f32<K>(col));
                const T sh = sg * shared[(size_t)t * (size_t)H + (size_t)(d0 + i)];
                out[(size_t)t * (size_t)H + (size_t)(d0 + i)] = r + sh;
            }
        }
        """

    nonisolated(unsafe) static let downCombineKernel = MLXFast.metalKernel(
        name: "track_moe_down_combine",
        inputNames: ["wd", "sd", "bd", "act", "idx", "w", "shared", "gate"],
        outputNames: ["out"],
        source: downCombineSource, header: helpers + TrackFastKernels.exactHeader + regHelpers,
        ensureRowContiguous: true)

    static func downCombine(
        wd: MLXArray, sd: MLXArray, bd: MLXArray, act: MLXArray, idx: MLXArray, w: MLXArray,
        shared: MLXArray, gate: MLXArray, topK: Int, groupSize: Int, bits: Int
    ) -> MLXArray {
        let BR = idx.dim(0), F = act.dim(1), H = wd.dim(1)
        let S = BR / topK
        precondition(BR % topK == 0 && H % 4 == 0 && bits == 4 && w.dtype == .float32)
        precondition(shared.dim(0) == S && shared.dim(1) == H && gate.dim(0) == S)
        return downCombineKernel(
            [wd, sd, bd, act, idx, w, shared, gate],
            template: [("T", act.dtype), ("GS", groupSize), ("BITS", bits), ("H", H), ("F", F), ("K", topK), ("FAST", isFast(k: F, n: H))],
            grid: (32, H / 4, S), threadGroup: (32, 1, 1),
            outputShapes: [[S, H]], outputDTypes: [act.dtype])[0]
    }
}
