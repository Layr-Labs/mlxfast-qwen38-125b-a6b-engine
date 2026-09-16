import Foundation
import MLX

/// Historical Qwen dual RMS adapted to Flash's *different* input widths:
/// embedding H=2560 and one statistic over the complete HC*H=10240 stream.
/// No concatenated H+H shortcut and no per-stream hidden normalization.
enum TrackDualRMS {
    private static let kernel = MLXFast.metalKernel(
        name: "track_flash_dual_rms_v3",
        inputNames: ["a", "b", "wa", "wb", "eps"],
        outputNames: ["ao", "bo"], source: """
            constexpr uint READS = 4;
            constexpr uint THREADS = 1024;
            const uint group = threadgroup_position_in_grid.x;
            const bool embedding = group < ROWS;
            const uint row = embedding ? group : group - ROWS;
            const uint width = embedding ? H : HC * H;
            const uint t = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const device T* x = (embedding ? a : b) + size_t(row) * width;
            const device T* w = embedding ? wa : wb;
            device T* y = (embedding ? ao : bo) + size_t(row) * width;
            float acc = 0.0f;
            for (uint start = 0; start < width; start += THREADS * READS) {
                const uint elem = start + t * READS;
                for (uint j = 0; j < READS; ++j) {
                    if (elem + j < width) {
                        const float xi = float(x[elem + j]);
                        acc += xi * xi;
                    }
                }
            }
            threadgroup float sums[32];
            acc = simd_sum(acc);
            if (lane == 0) { sums[sg] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc = simd_sum(sums[lane]);
            const float inv = metal::precise::rsqrt(acc / float(width) + eps);
            for (uint start = 0; start < width; start += THREADS * READS) {
                const uint elem = start + t * READS;
                for (uint j = 0; j < READS; ++j) {
                    if (elem + j < width) {
                        // RMS output rounds before BF16 norm-weight multiply.
                        y[elem + j] = w[elem + j] * T(float(x[elem + j]) * inv);
                    }
                }
            }
            """, ensureRowContiguous: true)

    static func apply(
        embedding: MLXArray, multi: MLXArray, embeddingWeight: MLXArray,
        hiddenWeight: MLXArray, eps: Float
    ) -> (embedding: MLXArray, multi: MLXArray)? {
        guard StreamOrDevice.default.stream === Stream.gpu,
            embedding.ndim == 3, multi.ndim == 3,
            embedding.dim(0) == 1, embedding.dim(1) >= 1, embedding.dim(1) <= 8,
            embedding.dim(2) == 2560, multi.shape == [1, embedding.dim(1), 10240],
            embedding.dtype == .bfloat16, multi.dtype == .bfloat16,
            embeddingWeight.shape == [2560], hiddenWeight.shape == [10240],
            embeddingWeight.dtype == .bfloat16, hiddenWeight.dtype == .bfloat16
        else { return nil }
        let out = kernel(
            [embedding, multi, embeddingWeight, hiddenWeight, eps],
            template: [("T", embedding.dtype), ("H", 2560), ("HC", 4), ("ROWS", embedding.dim(1))],
            grid: (2 * embedding.dim(1) * 1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [embedding.shape, multi.shape], outputDTypes: [.bfloat16, .bfloat16])
        return (out[0], out[1])
    }
}
