import Foundation
import MLX

enum TrackPairedAffine {
    static let packSource = #"""
        const uint i = thread_position_in_grid.x;
        if (i < (uint)COUNT) {
            packed[i] = uint(as_type<ushort>(scales[i]))
                | (uint(as_type<ushort>(biases[i])) << 16);
        }
        """#

    private static let packKernel = MLXFast.metalKernel(
        name: "track_pair_affine_metadata", inputNames: ["scales", "biases"],
        outputNames: ["packed"], source: packSource, ensureRowContiguous: true)

    static func pack(scales: MLXArray, biases: MLXArray) -> MLXArray {
        precondition(scales.dtype == .bfloat16 && biases.dtype == scales.dtype && scales.shape == biases.shape)
        return packKernel(
            [scales, biases], template: [("COUNT", scales.size)],
            grid: (scales.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [scales.shape], outputDTypes: [.uint32])[0]
    }

    static let helpers = #"""
        template <typename T, int group_size, int bits, int rows>
        METAL_FUNC void qmv_fast_reg_dual_paired(
            const device uint32_t* w0,
            const device uint32_t* metadata0,
            const device uint32_t* w1,
            const device uint32_t* metadata1,
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
          metadata0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          metadata1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          for (int k = 0; k < in_vec_size; k += block_size) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < rows; row++) {
              auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
              const auto pair0 = as_type<metal::vec<T, 2>>(metadata0[row * in_vec_size_g]);
              U s0 = pair0.x;
              U b0 = pair0.y;
              result0[row] += qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
              auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
              const auto pair1 = as_type<metal::vec<T, 2>>(metadata1[row * in_vec_size_g]);
              U s1 = pair1.x;
              U b1 = pair1.y;
              result1[row] += qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
            }
            ws0 += block_size * bytes_per_pack / pack_factor;
            ws1 += block_size * bytes_per_pack / pack_factor;
            metadata0 += block_size / group_size;
            metadata1 += block_size / group_size;
            x += block_size;
          }
          for (int row = 0; row < rows; row++) {
            result0[row] = simd_sum(result0[row]);
            result1[row] = simd_sum(result1[row]);
          }
        }
        """#

    static let source = #"""
        const uint z = threadgroup_position_in_grid.z;
        const bool shared = z == (uint)BR;
        const uint e = shared ? 0u : idx[z];
        const uint r = shared ? 0u : xrow[z];
        const size_t kw = (size_t)KD / 8;
        const size_t kg = (size_t)KD / GS;
        const size_t eoff = (size_t)e * (size_t)N;
        const device uint32_t* gw = shared ? wsh : wg + eoff * kw;
        const device uint32_t* gm = shared ? msh : mg + eoff * kg;
        const device uint32_t* uw = shared ? wsh + (size_t)N * kw : wu + eoff * kw;
        const device uint32_t* um = shared ? msh + (size_t)N * kg : mu + eoff * kg;
        const int out_row = (int)threadgroup_position_in_grid.y * (2 * RPS)
            + (int)simdgroup_index_in_threadgroup * RPS;
        float g[RPS], u[RPS];
        qmv_fast_reg_dual_paired<T, GS, BITS, RPS>(
            gw, gm, uw, um, x + (size_t)r * (size_t)KD,
            KD, out_row, thread_index_in_simdgroup, g, u);
        if (thread_index_in_simdgroup == 0) {
            for (int i = 0; i < RPS; ++i) {
                const T gv = static_cast<T>(g[i]);
                const T uv = static_cast<T>(u[i]);
                act[(size_t)z * (size_t)N + (size_t)(out_row + i)] = mlx_silu(gv) * uv;
            }
        }
        """#

    private static let kernel = MLXFast.metalKernel(
        name: "track_moe_gate_up_paired_metadata",
        inputNames: ["wg", "mg", "wu", "mu", "wsh", "msh", "x", "idx", "xrow"],
        outputNames: ["act"], source: source,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + helpers,
        ensureRowContiguous: true)

    static func gateUp(
        wg: MLXArray, wu: MLXArray, shared: TrackQuantWeight, metadata: [MLXArray],
        x: MLXArray, idx: MLXArray, xrow: MLXArray
    ) -> MLXArray {
        let rows = TrackFastMoEKernels.gateUpReuseRowsPerSimdgroup
        return kernel(
            [wg, metadata[0], wu, metadata[1], shared.weight, metadata[2], x, idx, xrow],
            template: [("T", x.dtype), ("GS", 32), ("BITS", 4), ("KD", 2560),
                       ("N", 640), ("BR", idx.size), ("RPS", rows)],
            grid: (32, 640 / rows, idx.size + 1), threadGroup: (32, 2, 1),
            outputShapes: [[idx.size + 1, 640]], outputDTypes: [x.dtype])[0]
    }
}
