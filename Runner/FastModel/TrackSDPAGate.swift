// Fused SDPA output gate for the canonical single-token attention path.
//
// The Metal bodies below are a narrow source-derived replica of the selected
// bfloat16/D=V=256 branches in:
// Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h
//
// Copyright © 2024 Apple Inc. for the source-derived SDPA bodies.
// MLX MIT License (Copyright © 2023 Apple Inc.):
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the “Software”),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

import Darwin
import MLX

enum TrackSDPAGate {
    private struct PartialResult {
        let partials: MLXArray
        let sums: MLXArray
        let maxs: MLXArray
        let blocks: Int

        init(partials: MLXArray, sums: MLXArray, maxs: MLXArray, blocks: Int) {
            self.partials = partials
            self.sums = sums
            self.maxs = maxs
            self.blocks = blocks
        }
    }

    private static let exactHeader = """
        template <typename T>
        METAL_FUNC T mlx_sigmoid(T x) {
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
        }
        """

    // Selected body from sdpa_vector_2pass_1. The host eligibility checks make
    // the native BF16/D256 paired branch unconditional here; no generic
    // fallback is reachable from this proposal.
    private static let partialSource = """
        const int N = keys_shape[2];
        const size_t k_head_stride = static_cast<size_t>(
            keys_shape[1] == 1 ? keys_strides[0] : keys_strides[1]);
        const size_t k_seq_stride = static_cast<size_t>(keys_strides[2]);
        const size_t v_head_stride = static_cast<size_t>(
            values_shape[1] == 1 ? values_strides[0] : values_strides[1]);
        const size_t v_seq_stride = static_cast<size_t>(values_strides[2]);

        const uint gqa = threads_per_threadgroup.y;
        const uint pair = thread_position_in_threadgroup.y;
        if (pair >= gqa / 2) { return; }
        const uint kv = threadgroup_position_in_grid.x;
        const uint batch = threadgroup_position_in_grid.y;
        const uint block = threadgroup_position_in_grid.z;
        const uint kv_batch = batch * threadgroups_per_grid.x + kv;
        const uint head0 = (batch * threadgroups_per_grid.x + kv) * gqa + pair * 2;
        const device InT* kp = keys + kv_batch * k_head_stride + block * k_seq_stride + thread_index_in_simdgroup * 8;
        const device InT* vp = values + kv_batch * v_head_stride + block * v_seq_stride + thread_index_in_simdgroup * 8;
        float4 q_lo[2], q_hi[2];
        float4 o_lo[2] = {float4(0), float4(0)};
        float4 o_hi[2] = {float4(0), float4(0)};
        float maximum[2] = {Limits<float>::finite_min, Limits<float>::finite_min};
        float denominator[2] = {0, 0};
        for (int h = 0; h < 2; ++h) {
            const device metal::vec<InT, 4>* qp =
                (const device metal::vec<InT, 4>*)(queries + (head0 + h) * D + thread_index_in_simdgroup * 8);
            q_lo[h] = static_cast<float>(scale[0]) * float4(qp[0]);
            q_hi[h] = static_cast<float>(scale[0]) * float4(qp[1]);
        }
        for (int token = static_cast<int>(block); token < N; token += BLOCKS) {
            const device metal::vec<InT, 4>* kv4 = (const device metal::vec<InT, 4>*)kp;
            const device metal::vec<InT, 4>* vv4 = (const device metal::vec<InT, 4>*)vp;
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
            kp += BLOCKS * int(k_seq_stride);
            vp += BLOCKS * int(v_seq_stride);
        }
        for (int h = 0; h < 2; ++h) {
            const uint offset = (head0 + h) * BLOCKS + block;
            if (thread_index_in_simdgroup == 0) {
                sums[offset] = denominator[h];
                maxs[offset] = maximum[h];
            }
            device metal::vec<InT, 4>* destination =
                (device metal::vec<InT, 4>*)(partials + offset * V + thread_index_in_simdgroup * 8);
            destination[0] = static_cast<metal::vec<InT, 4>>(o_lo[h]);
            destination[1] = static_cast<metal::vec<InT, 4>>(o_hi[h]);
        }
        """

    // Selected `batch_components` path from sdpa_vector_2pass_2. This is
    // limited to BF16 D=256, where the native scratch is 32*32*8 floats.
    // The final store keeps the native static_cast<BF16> before the optional
    // gate multiply.
    private static let finalSource = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = D / BD;
        typedef float U;

        thread U o[elem_per_thread] = {0};
        threadgroup U outputs[BN * BD * elem_per_thread];

        const int head_idx = threadgroup_position_in_grid.x;
        const int q_seq_idx = threadgroup_position_in_grid.y;
        const int q_offset = head_idx * threadgroups_per_grid.y + q_seq_idx;
        const int blocks = partials_shape[3];
        partials += q_offset * blocks * D + simdgroup_index_in_threadgroup * D +
            thread_index_in_simdgroup * elem_per_thread;
        sums += q_offset * blocks;
        maxs += q_offset * blocks;
        out += q_offset * D + simdgroup_index_in_threadgroup * elem_per_thread;

        U sum_exp_score = 0.0;
        U max_score = Limits<U>::finite_min;
        for (int b = 0; b < blocks / BN; ++b) {
            max_score = max(max_score, maxs[thread_index_in_simdgroup + BN * b]);
        }
        max_score = simd_max(max_score);
        for (int b = 0; b < blocks / BN; ++b) {
            U factor = fast::exp(maxs[thread_index_in_simdgroup + BN * b] - max_score);
            sum_exp_score += factor * sums[thread_index_in_simdgroup + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);
        for (int b = 0; b < blocks / BN; ++b) {
            U factor = fast::exp(maxs[simdgroup_index_in_threadgroup] - max_score);
            for (int i = 0; i < elem_per_thread; i++) {
                o[i] += factor * static_cast<U>(partials[i]);
            }
            maxs += BN;
            sums += BN;
            partials += BN * D;
        }

        for (int i = 0; i < elem_per_thread; i++) {
            outputs[i * BN * BD + thread_index_in_simdgroup * BD +
                simdgroup_index_in_threadgroup] = o[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] = simd_sum(outputs[i * BN * BD +
                simdgroup_index_in_threadgroup * BD + thread_index_in_simdgroup]);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        }

        if (thread_index_in_simdgroup == 0) {
            for (int i = 0; i < elem_per_thread; i++) {
                const InT a = static_cast<InT>(o[i]);
                if constexpr (APPLY_GATE) {
                    const uint d = simdgroup_index_in_threadgroup * elem_per_thread + i;
                    const size_t qkv_row = static_cast<size_t>(q_seq_idx) *
                        static_cast<size_t>(qkv_strides[1]);
                    const InT g = qkv[qkv_row + GATE_OFF + head_idx * D + d];
                    out[i] = a * mlx_sigmoid(g);
                } else {
                    out[i] = a;
                }
            }
        }
        """

    nonisolated(unsafe) private static let partialKernel = MLXFast.metalKernel(
        name: "track_sdpa_gate_partial",
        inputNames: ["queries", "keys", "values", "scale"],
        outputNames: ["partials", "sums", "maxs"],
        source: partialSource,
        header: exactHeader,
        ensureRowContiguous: false)

    nonisolated(unsafe) private static let finalKernel = MLXFast.metalKernel(
        name: "track_sdpa_gate_final",
        inputNames: ["partials", "sums", "maxs", "qkv"],
        outputNames: ["out"],
        source: finalSource,
        header: exactHeader,
        ensureRowContiguous: false)

    private static let architectureSuffix = GPU.deviceInfo().architecture.last

    static func blockCount(length: Int) -> Int? {
        guard (1024...2048).contains(length), getenv("MLX_SDPA_BLOCKS") == nil else { return nil }
        switch architectureSuffix {
        case "d": return 128
        case "s": return length == 1024 ? 64 : 128
        default: return nil
        }
    }

    private static func dispatchPartial(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: MLXArray,
        blocks: Int, stream: StreamOrDevice
    ) -> PartialResult {
        let outputs = partialKernel(
            [queries, keys, values, scale],
            template: [("InT", DType.bfloat16), ("D", 256), ("V", 256), ("BLOCKS", blocks)],
            grid: (64, 12, blocks), threadGroup: (32, 12, 1),
            outputShapes: [[1, 24, 1, blocks, 256], [1, 24, 1, blocks], [1, 24, 1, blocks]],
            outputDTypes: [.bfloat16, .float32, .float32], stream: stream)
        return PartialResult(partials: outputs[0], sums: outputs[1], maxs: outputs[2], blocks: blocks)
    }

    private static func dispatchFinish(
        _ partial: PartialResult, qkv: MLXArray, gated: Bool,
        stream: StreamOrDevice
    ) -> MLXArray {
        let outputs = finalKernel(
            [partial.partials, partial.sums, partial.maxs, qkv],
            template: [("InT", DType.bfloat16), ("D", 256),
                ("APPLY_GATE", gated), ("GATE_OFF", 6144)],
            grid: (24576, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [gated ? [1, 1, 6144] : [1, 24, 1, 256]],
            outputDTypes: [.bfloat16], stream: stream)
        return outputs[0]
    }

    /// Production path. The caller must enforce the canonical route
    /// and cache layout before calling this method; it performs no repeated
    /// host shape, dtype, or stride checks and returns only the fused output.
    static func apply(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: MLXArray,
        qkv: MLXArray, blocks: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        precondition(blocks == 64 || blocks == 128, "BLOCKS must be 64 or 128")
        let p = dispatchPartial(queries: queries, keys: keys, values: values, scale: scale,
            blocks: blocks, stream: stream)
        return dispatchFinish(p, qkv: qkv, gated: true, stream: stream)
    }

}
