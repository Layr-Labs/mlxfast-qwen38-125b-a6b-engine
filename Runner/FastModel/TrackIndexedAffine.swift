import Foundation
import MLX

struct IndexedAffineWords {
    let indices: [UInt16]
    let pairs: [UInt32]

    static func encode(scales: [UInt16], biases: [UInt16]) -> Self? {
        guard !scales.isEmpty, scales.count == biases.count else { return nil }
        var lookup: [UInt32: UInt16] = [:]
        lookup.reserveCapacity(8192)
        var pairs: [UInt32] = []
        pairs.reserveCapacity(8192)
        var indices = [UInt16](repeating: 0, count: scales.count)
        for i in scales.indices {
            let pair = UInt32(scales[i]) | (UInt32(biases[i]) << 16)
            if let index = lookup[pair] {
                indices[i] = index
            } else {
                guard pairs.count < 65536 else { return nil }
                let index = UInt16(pairs.count)
                lookup[pair] = index
                pairs.append(pair)
                indices[i] = index
            }
        }
        return Self(indices: indices, pairs: pairs)
    }
}

enum TrackIndexedAffine {
    static func cache(scales: MLXArray, biases: MLXArray) -> [MLXArray]? {
        guard scales.dtype == .bfloat16, biases.dtype == .bfloat16,
            scales.shape == biases.shape, scales.size > 0
        else { return nil }
        let s = scales.view(dtype: .uint16).asArray(UInt16.self)
        let b = biases.view(dtype: .uint16).asArray(UInt16.self)
        guard let encoded = IndexedAffineWords.encode(scales: s, biases: b) else { return nil }
        return [MLXArray(encoded.indices).reshaped(scales.shape), MLXArray(encoded.pairs)]
    }

    static let helpers = #"""
        template <typename T, int group_size, int bits, int rows>
        METAL_FUNC void qmv_fast_reg_dual_indexed(
            const device uint32_t* w0,
            const device ushort* metadata0,
            const device uint32_t* table0,
            const device uint32_t* w1,
            const device ushort* metadata1,
            const device uint32_t* table1,
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
              const auto pair0 = as_type<metal::vec<T, 2>>(table0[metadata0[row * in_vec_size_g]]);
              U s0 = pair0.x;
              U b0 = pair0.y;
              result0[row] += qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
              auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
              const auto pair1 = as_type<metal::vec<T, 2>>(table1[metadata1[row * in_vec_size_g]]);
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
        const device ushort* gm = shared ? msh : mg + eoff * kg;
        const device uint32_t* gt = shared ? lsh : lg;
        const device uint32_t* uw = shared ? wsh + (size_t)N * kw : wu + eoff * kw;
        const device ushort* um = shared ? msh + (size_t)N * kg : mu + eoff * kg;
        const device uint32_t* ut = shared ? lsh : lu;
        const int out_row = (int)threadgroup_position_in_grid.y * (2 * RPS)
            + (int)simdgroup_index_in_threadgroup * RPS;
        float g[RPS], u[RPS];
        qmv_fast_reg_dual_indexed<T, GS, BITS, RPS>(
            gw, gm, gt, uw, um, ut, x + (size_t)r * (size_t)KD,
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
        name: "track_moe_gate_up_indexed_metadata",
        inputNames: ["wg", "mg", "lg", "wu", "mu", "lu", "wsh", "msh", "lsh", "x", "idx", "xrow"],
        outputNames: ["act"], source: source,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + helpers,
        ensureRowContiguous: true)

    static func gateUp(
        wg: MLXArray, wu: MLXArray, shared: TrackQuantWeight, metadata: [MLXArray],
        x: MLXArray, idx: MLXArray, xrow: MLXArray
    ) -> MLXArray {
        let rows = TrackFastMoEKernels.gateUpReuseRowsPerSimdgroup
        return kernel(
            [wg, metadata[0], metadata[1], wu, metadata[2], metadata[3],
             shared.weight, metadata[4], metadata[5], x, idx, xrow],
            template: [("T", x.dtype), ("GS", 32), ("BITS", 4), ("KD", 2560),
                       ("N", 640), ("BR", idx.size), ("RPS", rows)],
            grid: (32, 640 / rows, idx.size + 1), threadGroup: (32, 2, 1),
            outputShapes: [[idx.size + 1, 640]], outputDTypes: [x.dtype])[0]
    }
}
