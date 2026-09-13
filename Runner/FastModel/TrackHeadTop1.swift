// TrackHeadTop1.swift -- fold shortlist top-1 into the drafter head GEMV.
//
// The host path is quantizedMM of the last sample against the shortlist rows,
// then argMax (lowest index wins) and take(ids). That materialises the full
// shortlist logits (~197 KiB bf16 / ~394 KiB f32) as a host-visible tensor.
// This kernel walks the same qmv_fast rows, keeps each 8-row tile's best in
// registers, reduces to a 16-byte packed record, and the host reads that.
// TRACK_HEAD_TOP1=0 restores the host read. The vendor token stream is
// unchanged: same GEMV arithmetic, same lowest-index tie-break.

import Foundation
import MLX

enum TrackHeadTop1 {
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_HEAD_TOP1"] ?? "1") != "0"
    }()

    /// Payload the host reads: token id, logit bits, shortlist index, byte count.
    static let packedBytes = 16
    static let packedWords = 4
    /// qmv_fast writes 8 rows per threadgroup (2 simdgroups × 4).
    static let tileRows = 8

    struct Selection: Equatable {
        var id: Int32
        var value: Float
        var index: Int
    }

    /// Linear scan, lowest index of the maximum. Same rule as MLX `argMax`.
    static func hostSelect(logits: [Float], ids: [Int32]) -> Selection {
        precondition(!logits.isEmpty && logits.count == ids.count)
        var bestI = 0
        var bestV = logits[0]
        for i in 1..<logits.count {
            if logits[i] > bestV {
                bestV = logits[i]
                bestI = i
            }
        }
        return Selection(id: ids[bestI], value: bestV, index: bestI)
    }

    /// Device reduction: best of each 8-row tile, then best of those tiles.
    /// Contiguous tiles plus lowest-index-wins make this equal to `hostSelect`.
    static func tiledSelect(logits: [Float], ids: [Int32]) -> Selection {
        precondition(logits.count == ids.count && logits.count % tileRows == 0)
        let tiles = logits.count / tileRows
        var tileVal = [Float](repeating: 0, count: tiles)
        var tileIdx = [Int](repeating: 0, count: tiles)
        for t in 0..<tiles {
            let base = t * tileRows
            var bi = base
            var bv = logits[base]
            for j in 1..<tileRows {
                let i = base + j
                if logits[i] > bv {
                    bv = logits[i]
                    bi = i
                }
            }
            tileVal[t] = bv
            tileIdx[t] = bi
        }
        var bestT = 0
        var bestV = tileVal[0]
        for t in 1..<tiles {
            if tileVal[t] > bestV {
                bestV = tileVal[t]
                bestT = t
            }
        }
        let index = tileIdx[bestT]
        return Selection(id: ids[index], value: bestV, index: index)
    }

    /// Decode the 16-byte packed record. `words` is 4 uint32s.
    static func unpack(_ words: [UInt32]) -> Selection {
        precondition(words.count == packedWords)
        return Selection(
            id: Int32(bitPattern: words[0]),
            value: Float(bitPattern: words[1]),
            index: Int(words[2]))
    }

    static func tokenId(_ packed: MLXArray) -> MLXArray {
        packed[0..<1].asType(.int32)
    }

    /// GEMV + tile argmax + reduce. Returns uint32[4] (16 bytes), or nil when
    /// the shape is not the qmv_fast shortlist (caller keeps the host path).
    static func apply(
        x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        ids: MLXArray, groupSize: Int, bits: Int
    ) -> MLXArray? {
        let k = x.dim(-1)
        let n = weight.dim(0)
        guard bits == 4, groupSize > 0, groupSize % 16 == 0,
            k % 512 == 0, n >= tileRows, n % tileRows == 0,
            x.size == k, weight.ndim == 2, weight.dtype == .uint32,
            weight.dim(1) == k / 8,
            scales.shape == [n, k / groupSize], biases.shape == scales.shape,
            scales.dtype == x.dtype, biases.dtype == x.dtype,
            x.dtype == .bfloat16 || x.dtype == .float32,
            ids.dtype == .int32, ids.size == n
        else { return nil }
        let xv = x.reshaped([k])
        let tiles = n / tileRows
        let gemv = gemvKernel(
            [xv, weight, scales, biases],
            template: [
                ("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("K", k), ("N", n),
            ],
            grid: (32, tiles * 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[tiles], [tiles]], outputDTypes: [.uint32, .float32])
        return reduceKernel(
            [gemv[0], gemv[1], ids.reshaped([n])],
            template: [("N", n)],
            grid: (32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[packedWords]], outputDTypes: [.uint32])[0]
    }

    static let qmvFastReg = #"""
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
        """#

    static let gemvSource = """
        const uint tile = threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        const int out_row = (int)tile * 8 + (int)sg * 4;
        float r[4];
        qmv_fast_reg<T, GS, BITS, 4>(w, scales, biases, x, K, out_row, lid, r);
        threadgroup float tv[8];
        threadgroup uint ti[8];
        if (lid == 0) {
            for (int i = 0; i < 4; ++i) {
                const T t = static_cast<T>(r[i]);
                tv[sg * 4 + i] = static_cast<float>(t);
                ti[sg * 4 + i] = (uint)(out_row + i);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lid == 0) {
            uint bi = ti[0];
            float bv = tv[0];
            for (int i = 1; i < 8; ++i) {
                if (tv[i] > bv) { bv = tv[i]; bi = ti[i]; }
            }
            pidx[tile] = bi;
            pval[tile] = bv;
        }
        """

    static let reduceSource = """
        const uint lane = thread_index_in_threadgroup;
        const uint ntiles = (uint)N / 8u;
        float bv = 0;
        uint bi = 0;
        uint have = 0;
        for (uint t = lane; t < ntiles; t += 32) {
            const float v = pval[t];
            const uint i = pidx[t];
            if (have == 0 || v > bv || (v == bv && i < bi)) {
                bv = v; bi = i; have = 1;
            }
        }
        for (uint off = 16; off > 0; off /= 2) {
            const float nv = simd_shuffle_down(bv, off);
            const uint ni = simd_shuffle_down(bi, off);
            const uint nh = simd_shuffle_down(have, off);
            if (nh != 0 && (have == 0 || nv > bv || (nv == bv && ni < bi))) {
                bv = nv; bi = ni; have = 1;
            }
        }
        if (lane == 0) {
            packed[0] = uint(ids[bi]);
            packed[1] = as_type<uint>(bv);
            packed[2] = bi;
            packed[3] = 16u;
        }
        """

    nonisolated(unsafe) static let gemvKernel = MLXFast.metalKernel(
        name: "track_head_top1_gemv",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["pidx", "pval"],
        source: gemvSource,
        header: TrackFastMoEKernels.helpersCore + qmvFastReg,
        ensureRowContiguous: true)

    nonisolated(unsafe) static let reduceKernel = MLXFast.metalKernel(
        name: "track_head_top1_reduce",
        inputNames: ["pidx", "pval", "ids"],
        outputNames: ["packed"],
        source: reduceSource, ensureRowContiguous: true)
}
