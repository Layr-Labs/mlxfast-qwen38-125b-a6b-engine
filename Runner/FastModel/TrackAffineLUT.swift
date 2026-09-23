// Load-time derived index for the affine metadata of 4-bit tensors.
//
// A 4-bit affine tensor streams, per group of 32 weights, 16 bytes of nibbles
// plus one bf16 scale and one bf16 bias (20 bytes). Measured on the pinned
// checkpoint, each tensor carries only a few thousand distinct (scale, bias)
// bit pairs, so the pair can be named by a 16-bit index into a per-tensor
// table that stays cache resident: 18 bytes per group instead of 20, i.e.
// -10% of the bytes a one-token GEMV streams from DRAM.
//
// The checkpoint is not touched: the index and the table are built in memory
// at load from the loaded bf16 `scales` / `biases` (an input-independent
// derived cache, like a dequantized copy), and every group is verified to
// reconstruct its exact bit pair before the table is used. The kernels that
// read through the table convert the SAME bf16 bit patterns to float that the
// reference kernels read, so every product, per-lane accumulation and
// `simd_sum` sees identical operands in identical order.

import Foundation
import MLX
import MLXFast

struct TrackAffineLUT {
    /// uint16, same shape as `scales`: the pair index of each group.
    let index: MLXArray
    /// uint32 `[count]`: `scale_bits | bias_bits << 16` of each distinct pair.
    let table: MLXArray
    let count: Int

    /// Runtime switch (env `TRACK_AFFINE_LUT=0` keeps the plain bf16 path).
    nonisolated(unsafe) static var enabled: Bool =
        ProcessInfo.processInfo.environment["TRACK_AFFINE_LUT"] != "0"
    /// Tables above this size are not built (they would not stay cache resident).
    static let maxEntries = 32768

    /// Build the index for one tensor, or nil when the tensor does not qualify
    /// (dtype, size) or the round trip does not reproduce every pair exactly.
    static func build(scales: MLXArray, biases: MLXArray?) -> TrackAffineLUT? {
        guard enabled, let biases, scales.dtype == .bfloat16, biases.dtype == .bfloat16,
            scales.shape == biases.shape, scales.size > 0
        else { return nil }
        let pairs = (scales.view(dtype: .uint16).asType(.uint32) | (biases.view(dtype: .uint16).asType(.uint32) << 16))
        if scales.size >= 1 << 20 { return buildOnDevice(pairs: pairs, shape: scales.shape) }
        let host = pairs.reshaped(-1).asArray(UInt32.self)
        var pairToIndex: [UInt32: UInt16] = [:]
        pairToIndex.reserveCapacity(8192)
        var table: [UInt32] = []
        table.reserveCapacity(8192)
        var index = [UInt16](repeating: 0, count: host.count)
        for (i, p) in host.enumerated() {
            if let j = pairToIndex[p] {
                index[i] = j
            } else {
                if table.count >= maxEntries { return nil }
                let j = UInt16(table.count)
                pairToIndex[p] = j
                table.append(p)
                index[i] = j
            }
        }
        let indexArray = MLXArray(index).reshaped(scales.shape)
        let tableArray = MLXArray(table)
        return verified(index: indexArray, table: tableArray, count: table.count, pairs: pairs)
    }

    /// The same index built on the GPU (sort, run boundaries, scatter): used
    /// for the large expert stacks, where the host loop would take seconds.
    static func buildOnDevice(pairs: MLXArray, shape: [Int]) -> TrackAffineLUT? {
        let flat = pairs.reshaped(-1)
        let G = flat.size
        let order = argSort(flat).asType(.int32)
        let sorted = flat[order]
        let isNew = concatenated([MLXArray([Int32(1)]), (sorted[1...] .!= sorted[..<(G - 1)]).asType(.int32)])
        let rank = cumsum(isNew) - 1
        let count = Int(rank[G - 1].item(Int32.self)) + 1
        guard count <= maxEntries else { return nil }
        var index = MLXArray.zeros([G], dtype: .int32)
        index[order] = rank
        var table = MLXArray.zeros([count], dtype: .uint32)
        table[rank] = sorted  // duplicate positions write the same value
        return verified(index: index.asType(.uint16).reshaped(shape), table: table, count: count, pairs: pairs)
    }

    /// Fail closed: every group must reconstruct its exact (scale, bias) bits.
    private static func verified(index: MLXArray, table: MLXArray, count: Int, pairs: MLXArray) -> TrackAffineLUT? {
        let back = table[index.asType(.int32)]
        guard (back .== pairs).all().item(Bool.self) else { return nil }
        eval(index, table)
        return TrackAffineLUT(index: index, table: table, count: count)
    }
}

/// One-token GEMV over a 4-bit affine tensor whose metadata is read through
/// the pair table: MLX's `qmv_fast_impl` / `qmv_impl` (the kernels the
/// reference dispatches for M = 1) with the two bf16 reads per group replaced
/// by one index read and a table lookup. Same lanes, same block walk, same
/// per-lane accumulation, same `simd_sum`.
enum TrackLUTGemv {
    static let twins = #"""
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_impl_lut(
    const device uint32_t* w,
    const device ushort* sidx,
    const device uint* lut,
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
  sidx += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device ushort* il = sidx + row * in_vec_size_g;

      const uint p = lut[il[0]];
      U s = as_type<bfloat16_t>(ushort(p & 0xffffu));
      U b = as_type<bfloat16_t>(ushort(p >> 16));
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    sidx += block_size / group_size;
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
METAL_FUNC void qmv_impl_lut(
    const device uint32_t* w,
    const device ushort* sidx,
    const device uint* lut,
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
    sidx += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; out_row + row < out_vec_size; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device ushort* il = sidx + row * in_vec_size_g;

        const uint p = lut[il[0]];
        U s = as_type<bfloat16_t>(ushort(p & 0xffffu));
        U b = as_type<bfloat16_t>(ushort(p >> 16));
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      sidx += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; out_row + row < out_vec_size; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device ushort* il = sidx + row * in_vec_size_g;

        const uint p = lut[il[0]];
        U s = as_type<bfloat16_t>(ushort(p & 0xffffu));
        U b = as_type<bfloat16_t>(ushort(p >> 16));
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; out_row + row < out_vec_size; row++) {
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
    sidx += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device ushort* il = sidx + row * in_vec_size_g;

        const uint p = lut[il[0]];
        U s = as_type<bfloat16_t>(ushort(p & 0xffffu));
        U b = as_type<bfloat16_t>(ushort(p >> 16));
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      sidx += block_size / group_size;
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
        const device ushort* il = sidx + row * in_vec_size_g;

        const uint p = lut[il[0]];
        U s = as_type<bfloat16_t>(ushort(p & 0xffffu));
        U b = as_type<bfloat16_t>(ushort(p >> 16));
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
"""#

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_qmv_lut",
        inputNames: ["w", "sidx", "lut", "x"],
        outputNames: ["y"],
        source: """
            if (FAST) {
                qmv_fast_impl_lut<T, GS, BITS>(w, sidx, lut, x, y, K, N, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            } else {
                qmv_impl_lut<T, GS, BITS>(w, sidx, lut, x, y, K, N, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            }
            """,
        header: TrackFastMoEKernels.helpersCore + twins, ensureRowContiguous: true)

    /// `x [..., 1, K]` (one row) -> `[..., 1, N]`, or nil when the shape is not
    /// the one-row GEMV the reference serves with `qmv_fast` / `qmv`.
    static func apply(_ x: MLXArray, weight w: MLXArray, meta: TrackAffineLUT, groupSize: Int, bits: Int) -> MLXArray? {
        guard bits == 4, x.ndim >= 2, x.dim(-2) == 1, x.dtype == .bfloat16 else { return nil }
        let K = x.dim(-1), N = w.dim(0)
        guard K == w.dim(1) * 32 / bits, K % groupSize == 0 else { return nil }
        let batch = x.size / K
        guard batch == 1 else { return nil }
        let fast = N % 8 == 0 && K % 512 == 0
        let tgY = (N + 7) / 8
        let y = kernel(
            [w, meta.index, meta.table, x.reshaped(K)],
            template: [("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("K", K), ("N", N), ("FAST", fast)],
            grid: (32, tgY * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[N]], outputDTypes: [x.dtype])[0]
        var shape = x.shape
        shape[shape.count - 1] = N
        return y.reshaped(shape)
    }
}
