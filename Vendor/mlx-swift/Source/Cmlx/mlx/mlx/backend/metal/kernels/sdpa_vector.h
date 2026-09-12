// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Two query heads reuse each K/V load while retaining their own block walk.
  // K/V/Q move in 16-byte vector loads; the score keeps its sequential
  // component order so the accumulated value is unchanged.
  if constexpr (metal::is_same_v<T, bfloat16_t> && D == 256 && V == 256) {
    const uint gqa = tptg.y;
    if (N >= 1024 && tptg.z == 1 && gqa % 2 == 0 && !has_mask && !has_sinks) {
      const uint pair = tidtg.y;
      if (pair >= gqa / 2) { return; }
      const uint kv = tid.x;
      const uint batch = tid.y;
      const uint block = tid.z;
      const uint kv_batch = batch * tpg.x + kv;
      const uint head0 = (batch * tpg.x + kv) * gqa + pair * 2;
      const device T* kp = keys + kv_batch * k_head_stride + block * k_seq_stride + simd_lid * 8;
      const device T* vp = values + kv_batch * v_head_stride + block * v_seq_stride + simd_lid * 8;
      float4 q_lo[2], q_hi[2];
      float4 o_lo[2] = {float4(0), float4(0)};
      float4 o_hi[2] = {float4(0), float4(0)};
      float maximum[2] = {Limits<float>::finite_min, Limits<float>::finite_min};
      float denominator[2] = {0, 0};
      for (int h = 0; h < 2; ++h) {
        const device metal::vec<T, 4>* qp =
            (const device metal::vec<T, 4>*)(queries + (head0 + h) * D + simd_lid * 8);
        q_lo[h] = static_cast<float>(scale) * float4(qp[0]);
        q_hi[h] = static_cast<float>(scale) * float4(qp[1]);
      }
      // Several tokens per iteration: their loads and dot products are
      // independent of each other and of the accumulate chain, so they overlap
      // the simd_sum latency. Each head still consumes its tokens in order.
      constexpr int kTokensPerIter = 2;
      const int k_step = blocks * int(k_seq_stride);
      const int v_step = blocks * int(v_seq_stride);
      int token = block;
      for (; token + (kTokensPerIter - 1) * blocks < N;
           token += kTokensPerIter * blocks) {
        float4 k_lo[kTokensPerIter], k_hi[kTokensPerIter];
        float4 v_lo[kTokensPerIter], v_hi[kTokensPerIter];
        for (int u = 0; u < kTokensPerIter; ++u) {
          const device metal::vec<T, 4>* kk =
              (const device metal::vec<T, 4>*)(kp + u * k_step);
          const device metal::vec<T, 4>* vv =
              (const device metal::vec<T, 4>*)(vp + u * v_step);
          k_lo[u] = float4(kk[0]);
          k_hi[u] = float4(kk[1]);
          v_lo[u] = float4(vv[0]);
          v_hi[u] = float4(vv[1]);
        }
        float score[2][kTokensPerIter];
        for (int h = 0; h < 2; ++h) {
          for (int u = 0; u < kTokensPerIter; ++u) {
            float s = q_lo[h].x * k_lo[u].x;
            s += q_lo[h].y * k_lo[u].y;
            s += q_lo[h].z * k_lo[u].z;
            s += q_lo[h].w * k_lo[u].w;
            s += q_hi[h].x * k_hi[u].x;
            s += q_hi[h].y * k_hi[u].y;
            s += q_hi[h].z * k_hi[u].z;
            s += q_hi[h].w * k_hi[u].w;
            score[h][u] = simd_sum(s);
          }
        }
        for (int h = 0; h < 2; ++h) {
          for (int u = 0; u < kTokensPerIter; ++u) {
            const float next_maximum = max(maximum[h], score[h][u]);
            const float factor = fast::exp(maximum[h] - next_maximum);
            const float exp_score = fast::exp(score[h][u] - next_maximum);
            maximum[h] = next_maximum;
            denominator[h] = denominator[h] * factor + exp_score;
            o_lo[h] = o_lo[h] * factor + exp_score * v_lo[u];
            o_hi[h] = o_hi[h] * factor + exp_score * v_hi[u];
          }
        }
        kp += kTokensPerIter * k_step;
        vp += kTokensPerIter * v_step;
      }
      for (; token < N; token += blocks) {
        const device metal::vec<T, 4>* kv4 = (const device metal::vec<T, 4>*)kp;
        const device metal::vec<T, 4>* vv4 = (const device metal::vec<T, 4>*)vp;
        const float4 k_lo = float4(kv4[0]);
        const float4 k_hi = float4(kv4[1]);
        const float4 v_lo = float4(vv4[0]);
        const float4 v_hi = float4(vv4[1]);
        for (int h = 0; h < 2; ++h) {
          float score = q_lo[h].x * k_lo.x;
          score += q_lo[h].y * k_lo.y;
          score += q_lo[h].z * k_lo.z;
          score += q_lo[h].w * k_lo.w;
          score += q_hi[h].x * k_hi.x;
          score += q_hi[h].y * k_hi.y;
          score += q_hi[h].z * k_hi.z;
          score += q_hi[h].w * k_hi.w;
          score = simd_sum(score);
          const float next_maximum = max(maximum[h], score);
          const float factor = fast::exp(maximum[h] - next_maximum);
          const float exp_score = fast::exp(score - next_maximum);
          maximum[h] = next_maximum;
          denominator[h] = denominator[h] * factor + exp_score;
          o_lo[h] = o_lo[h] * factor + exp_score * v_lo;
          o_hi[h] = o_hi[h] * factor + exp_score * v_hi;
        }
        kp += k_step;
        vp += v_step;
      }
      for (int h = 0; h < 2; ++h) {
        const uint offset = (head0 + h) * blocks + block;
        if (simd_lid == 0) {
          sums[offset] = denominator[h];
          maxs[offset] = maximum[h];
        }
        device metal::vec<T, 4>* destination =
            (device metal::vec<T, 4>*)(out + offset * V + simd_lid * 8);
        destination[0] = static_cast<metal::vec<T, 4>>(o_lo[h]);
        destination[1] = static_cast<metal::vec<T, 4>>(o_hi[h]);
      }
      return;
    }
  }

  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Duplication-free variant for high gqa_factor decode: each simdgroup owns a
// contiguous token sub-chunk and computes HPT of its group's query heads, so
// each K/V byte is read G / HPT times instead of G times. Single-token
// queries without mask or sinks only; the partials layout matches
// sdpa_vector_2pass_2.
template <typename T, int D, int V, int G, int HPT>
[[kernel]] void sdpa_vector_2pass_1_gqa(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int NT = G / HPT;

  typedef float U;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int blocks = tpg.z;
  const int g = tidtg.y;
  const int cchunk = g / NT;
  const int h0 = (g % NT) * HPT;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * G;
  const int base_head = batch_idx * num_q_heads + kv_head_idx * G;

  const int chunk = (N + blocks - 1) / blocks;
  const int kstart = block_idx * chunk;
  const int kend = min(N, kstart + chunk);
  const int sub = (chunk + HPT - 1) / HPT;
  const int s0 = kstart + cchunk * sub;
  const int s1 = min(kend, s0 + sub);

  const device T* kp = keys + kv_head_idx * k_head_stride + s0 * k_seq_stride +
      simd_lid * qk_per_thread;
  const device T* vp = values + kv_head_idx * v_head_stride +
      s0 * v_seq_stride + simd_lid * v_per_thread;

  U q[HPT][qk_per_thread];
  for (int j = 0; j < HPT; j++) {
    const device T* qp =
        queries + (base_head + h0 + j) * D + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[j][i] = static_cast<U>(scale) * qp[i];
    }
  }

  U max_score[HPT];
  U sum_exp_score[HPT];
  U o[HPT][v_per_thread];
  for (int j = 0; j < HPT; j++) {
    max_score[j] = Limits<U>::finite_min;
    sum_exp_score[j] = 0;
    for (int i = 0; i < v_per_thread; i++) {
      o[j][i] = 0;
    }
  }

  for (int t = s0; t < s1; t++) {
    U kr[qk_per_thread];
    U vr[v_per_thread];
    for (int i = 0; i < qk_per_thread; i++) {
      kr[i] = kp[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      vr[i] = vp[i];
    }
    kp += k_seq_stride;
    vp += v_seq_stride;
    for (int j = 0; j < HPT; j++) {
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[j][i] * kr[i];
      }
      score = simd_sum(score);
      U new_max = max(max_score[j], score);
      U factor = fast::exp(max_score[j] - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score[j] = new_max;
      sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;
      for (int i = 0; i < v_per_thread; i++) {
        o[j][i] = o[j][i] * factor + exp_score * vr[i];
      }
    }
  }

  threadgroup U o_sh[G * HPT * V];
  threadgroup U se_sh[G * HPT];
  threadgroup U mx_sh[G * HPT];
  for (int j = 0; j < HPT; j++) {
    int slot = (h0 + j) * HPT + cchunk;
    U inv = sum_exp_score[j] > 0 ? 1 / sum_exp_score[j] : 0;
    for (int i = 0; i < v_per_thread; i++) {
      o_sh[slot * V + simd_lid * v_per_thread + i] = o[j][i] * inv;
    }
    if (simd_lid == 0) {
      se_sh[slot] = sum_exp_score[j];
      mx_sh[slot] = max_score[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U gmax = Limits<U>::finite_min;
  for (int s = 0; s < HPT; s++) {
    gmax = max(gmax, mx_sh[g * HPT + s]);
  }
  U denom = 0;
  U acc[v_per_thread] = {0};
  for (int s = 0; s < HPT; s++) {
    U w = se_sh[g * HPT + s] * fast::exp(mx_sh[g * HPT + s] - gmax);
    denom += w;
    for (int i = 0; i < v_per_thread; i++) {
      acc[i] += w * o_sh[(g * HPT + s) * V + simd_lid * v_per_thread + i];
    }
  }

  const int o_offset = base_head + g;
  device T* op =
      out + o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  for (int i = 0; i < v_per_thread; i++) {
    op[i] = static_cast<T>(acc[i]);
  }
  if (simd_lid == 0) {
    sums[o_offset * blocks + block_idx] = denom;
    maxs[o_offset * blocks + block_idx] = gmax;
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  constexpr bool batch_components = metal::is_same_v<T, bfloat16_t> && D == 256;
  threadgroup U outputs[BN * BD * (batch_components ? elem_per_thread : 1)];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  if constexpr (batch_components) {
    for (int i = 0; i < elem_per_thread; i++) {
      outputs[i * BN * BD + simd_lid * BD + simd_gid] = o[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = simd_sum(outputs[i * BN * BD + simd_gid * BD + simd_lid]);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
  } else {
    for (int i = 0; i < elem_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
