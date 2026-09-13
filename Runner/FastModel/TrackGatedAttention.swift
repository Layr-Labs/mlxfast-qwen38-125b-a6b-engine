import Foundation
import MLX
import MLXLLM
import MLXLMCommon

enum TrackGatedAttention {
    private static let supportedDevice = GPU.deviceInfo().architecture == "applegpu_g17s"
    private static let defaultBlocks = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"] == nil

    private static let singleKernel = MLXFast.metalKernel(
        name: "track_attention_gated_single",
        inputNames: ["queries", "keys", "values", "gates"], outputNames: ["out"],
        source: singleSource, header: TrackFastKernels.exactHeader, ensureRowContiguous: false)
    private static let firstKernel = MLXFast.metalKernel(
        name: "track_attention_gated_partials",
        inputNames: ["queries", "keys", "values"], outputNames: ["out", "sums", "maxs"],
        source: firstSource, header: "", ensureRowContiguous: false)
    private static let finalKernel = MLXFast.metalKernel(
        name: "track_attention_gated_final",
        inputNames: ["partials", "sums", "maxs", "gates"], outputNames: ["out"],
        source: finalSource, header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static func apply(
        queries: MLXArray, cache: Qwen4ExpCBv2LayerCache, gates: MLXArray,
        gateOffset: Int, scale: Float
    ) -> MLXArray? {
        guard StreamOrDevice.default.stream == Stream.gpu, supportedDevice, defaultBlocks,
            queries.shape == [1, 24, 1, 256], queries.dtype == .bfloat16,
            queries.strides[1] == 256, queries.strides[3] == 1,
            gates.ndim == 3, gates.dim(0) == 1, gates.dim(1) == 1,
            gates.dtype == .bfloat16, gates.strides[2] == 1,
            gateOffset >= 0, gateOffset + 6144 <= gates.dim(2),
            scale == 0.0625, cache.rows.count == 1,
            cache.kind.attention == .full, !cache.kind.hasSinks, !cache.kind.isBidirectional,
            let row = cache.rows[0] as? CBv2FullSequenceKV,
            row.absoluteOffset >= 1, row.absoluteOffset <= 2048
        else { return nil }
        let snapshot = row.snapshot()
        let keys = snapshot.keys, values = snapshot.values
        let count = row.absoluteOffset
        guard keys.shape == [1, 2, count, 256], values.shape == keys.shape,
            keys.dtype == .bfloat16, values.dtype == .bfloat16,
            keys.strides[3] == 1, values.strides[3] == 1,
            keys.strides[1] > 0, keys.strides[2] >= 256,
            values.strides[1] > 0, values.strides[2] >= 256
        else { return nil }
        if count < 1024 {
            return singleKernel(
                [queries, keys, values, gates],
                template: [("T", queries.dtype), ("D", 256), ("V", 256), ("GATE_OFF", gateOffset)],
                grid: (24 * 1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 6144]], outputDTypes: [.bfloat16])[0]
        }
        let blocks = count == 1024 ? 64 : 128
        let partials = firstKernel(
            [queries, keys, values],
            template: [("T", queries.dtype), ("D", 256), ("V", 256), ("BLOCKS", blocks)],
            grid: (2 * 32, 12, blocks), threadGroup: (32, 12, 1),
            outputShapes: [[24, blocks, 256], [24, blocks], [24, blocks]],
            outputDTypes: [.bfloat16, .float32, .float32])
        return finalKernel(
            [partials[0], partials[1], partials[2], gates],
            template: [("T", queries.dtype), ("D", 256), ("BLOCKS", blocks), ("GATE_OFF", gateOffset)],
            grid: (24 * 1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, 1, 6144]], outputDTypes: [.bfloat16])[0]
    }
}

// Copyright © 2024 Apple Inc.
extension TrackGatedAttention {
    static let singleSource = #"""
        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        constexpr bool has_mask=false, query_transposed=false, do_causal=false;
        constexpr bool bool_mask=false, float_mask=false, has_sinks=false;
        constexpr int mask_kv_seq_stride=0, mask_q_seq_stride=0, mask_head_stride=0;
        const device bool* bmask=nullptr;
        const device T* fmask=nullptr;
        const device T* sinks=nullptr;
        const int N=keys_shape[2];
        const size_t k_head_stride=keys_strides[1], k_seq_stride=keys_strides[2];
        const size_t v_head_stride=values_strides[1], v_seq_stride=values_strides[2];
        constexpr float scale=0.0625f;
        constexpr int gqa_factor=12, num_q_heads=24;
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
              out[i] = static_cast<T>(o[i]) * mlx_sigmoid(gates[GATE_OFF + tid.x * D + simd_gid * v_per_thread + i]);
            }
          }
        """#
    static let firstSource = #"""
        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        constexpr bool has_mask=false, query_transposed=false, do_causal=false;
        constexpr bool bool_mask=false, float_mask=false, has_sinks=false;
        constexpr int mask_kv_seq_stride=0, mask_q_seq_stride=0, mask_head_stride=0;
        const device bool* bmask=nullptr;
        const device T* fmask=nullptr;
        const device T* sinks=nullptr;
        const int N=keys_shape[2];
        const size_t k_head_stride=keys_strides[1], k_seq_stride=keys_strides[2];
        const size_t v_head_stride=values_strides[1], v_seq_stride=values_strides[2];
        constexpr float scale=0.0625f;
        const uint3 tptg=threads_per_threadgroup;
        const uint3 tidtg=thread_position_in_threadgroup;
        constexpr int blocks=BLOCKS;
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
              for (int token = block; token < N; token += blocks) {
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
                kp += blocks * int(k_seq_stride);
                vp += blocks * int(v_seq_stride);
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
        """#
    static let finalSource = #"""
        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        constexpr int blocks=BLOCKS;
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
              out[i] = static_cast<T>(o[i]) * mlx_sigmoid(gates[GATE_OFF + tid.x * D + simd_gid * elem_per_thread + i]);
            }
          }
        """#
}
