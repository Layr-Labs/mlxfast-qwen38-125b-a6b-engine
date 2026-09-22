// One-token GDN input projection. Same 4-bit qdot order as MLX qmv_fast.
// Eight simdgroups share one threadgroup copy of the 2560-wide activation.
// Each simdgroup still walks four rows, in the same block order, and stores
// from lane 0. Every other shape stays on TrackMultiProj.apply.

import Foundation
import MLX

enum TrackFastGDNProj {
    static let source = """
        constexpr int kGroups = 8;
        constexpr int kRows = 4;
        constexpr int kValues = 16;
        constexpr int kBlock = 512;
        const int tile = int(threadgroup_position_in_grid.y);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int lid = int(thread_index_in_simdgroup);
        threadgroup T act[K];
        const int linear = sg * 32 + lid;
        for (int i = linear; i < K; i += kGroups * 32) {
            act[i] = x[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int out_row = tile * (kGroups * kRows) + sg * kRows;
        constexpr int pack_factor = 8;
        constexpr int bytes_per_pack = 4;
        constexpr int packs_per_thread = 2;
        constexpr int scale_step = 2;
        const int row_bytes = K * bytes_per_pack / pack_factor;
        const int row_groups = K / 32;
        const device uint8_t* ws = ((const device uint8_t*)w)
            + out_row * row_bytes
            + lid * packs_per_thread * bytes_per_pack;
        const device T* sp = scales + out_row * row_groups + lid / scale_step;
        const device T* bp = biases + out_row * row_groups + lid / scale_step;
        const threadgroup T* xp = act + lid * kValues;
        float result[kRows] = {0, 0, 0, 0};
        for (int k = 0; k < K; k += kBlock) {
            float x_thread[kValues];
            float sum = 0;
            for (int i = 0; i < kValues; i += 4) {
                sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3];
                x_thread[i] = xp[i];
                x_thread[i + 1] = xp[i + 1] / 16.0f;
                x_thread[i + 2] = xp[i + 2] / 256.0f;
                x_thread[i + 3] = xp[i + 3] / 4096.0f;
            }
            for (int row = 0; row < kRows; row++) {
                const device uint8_t* wl = ws + row * row_bytes;
                const float s = sp[row * row_groups];
                const float b = bp[row * row_groups];
                result[row] += qdot<float, kValues, 4>(wl, x_thread, s, b, sum);
            }
            ws += kBlock * bytes_per_pack / pack_factor;
            sp += kBlock / 32;
            bp += kBlock / 32;
            xp += kBlock;
        }
        for (int row = 0; row < kRows; row++) {
            result[row] = simd_sum(result[row]);
            if (lid == 0) {
                y[out_row + row] = static_cast<T>(result[row]);
            }
        }
        """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_gdn_in_proj",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: source,
        header: TrackFastMoEKernels.helpersCore,
        ensureRowContiguous: true)

    /// One-token fused GDN input projection. Nil means the caller uses `apply`.
    static func y(_ x: MLXArray, multi: TrackMultiProj) -> MLXArray? {
        guard StreamOrDevice.default.stream === Stream.gpu,
            x.dim(-1) == 2560,
            x.shape.dropLast().reduce(1, *) == 1,
            x.dtype == .bfloat16,
            let fused = multi.fused,
            case .quant(let q) = fused,
            q.bits == 4,
            q.groupSize == 32,
            q.mode == .affine,
            let biases = q.biases,
            q.rows == 16480,
            q.weight.dim(1) == 320,
            biases.dim(0) == 16480
        else { return nil }
        let rows = q.rows
        let y = kernel(
            [q.weight, q.scales, biases, x.reshaped(2560)],
            template: [("T", x.dtype), ("K", 2560)],
            grid: (32, (rows / 32) * 8, 1),
            threadGroup: (32, 8, 1),
            outputShapes: [[rows]],
            outputDTypes: [x.dtype])[0]
        return y.reshaped(x.shape.dropLast() + [rows])
    }
}
