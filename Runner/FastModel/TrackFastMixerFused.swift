// Hyper-connection mixer, norm folded in: for a one-token window the
// `track_inject_norm` launch and the `track_mixer_down_inject` launch become
// one kernel (`track_norm_down_inject`).
//
// Why it can be one launch at all. The norm's `normed` row (HC * H = 10240
// values) is consumed in FULL by every row of the down and inject GEMVs, so the
// two launches are separated by a grid-wide dependency that no tiling removes.
// The fusion pays for that dependency instead of synchronising on it: every
// threadgroup recomputes the whole norm into its own threadgroup buffer, and
// the GEMV walk then reads the vector out of threadgroup memory. The redundant
// pass is 45 KB of reads per threadgroup over data the last launch just
// touched; the dispatch it removes is ~10 us of fixed cost, 97 times a step.
//
// Why it is bit-identical. Nothing about the arithmetic moves:
//  * the RMS reduction replays `track_inject_norm`'s 640-thread layout exactly
//    the way `track_inject_norm_wide` already does -- lane l of iteration g
//    plays virtual thread 32g + l, so every `simd_sum` sees the same values in
//    the same lanes, the per-simdgroup partials land in the same
//    `local_sums[g]`, and the closing 32-lane `simd_sum` and
//    `precise::rsqrt(acc / H + eps)` are the same instructions on the same bits;
//  * `stream = residual + out * inject` keeps its bf16 multiply and add;
//  * `normed = bf16(f32(stream) * inv_mean) * scale` keeps its single rounding
//    before the weight;
//  * the down/inject walks are `qmv_fast_reg` / `track_inject_qmv_row`
//    character for character, with the vector pointer moved from the device to
//    the threadgroup address space. Which simdgroup owns which output row
//    changes; what a row's owner computes does not.
//
// Windows of two or more tokens keep the unfused pair.

import Foundation
import MLX
import MLXFast

enum TrackFastFusedMixer {

    /// `MLXFAST_HC_FUSE=0` routes every mixer back to the two-launch pair; the
    /// same binary then A/Bs the fusion. Read once.
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["MLXFAST_HC_FUSE"] ?? "1") != "0"
    }()
    /// Simdgroups per threadgroup, at least HC. Twenty is the shipped choice:
    /// the norm layout has exactly 20 simdgroups' worth of virtual threads per
    /// stream, so at NSG = 20 every thread replays exactly one of them and the
    /// redundant pass is one iteration deep. Fewer simdgroups make that pass
    /// two or three iterations deep, which is what the sweep charges for.
    nonisolated(unsafe) static var simdgroups: Int = {
        ProcessInfo.processInfo.environment["MLXFAST_HC_FUSE_SG"].flatMap { Int($0) } ?? 20
    }()
    /// Down rows owned by one simdgroup.
    nonisolated(unsafe) static var rowsPerSimdgroup: Int = {
        ProcessInfo.processInfo.environment["MLXFAST_HC_FUSE_ROWS"].flatMap { Int($0) } ?? 1
    }()

    /// Threadgroup-address-space copies of the two vector loads and the two
    /// GEMV walks. Only the address space of `x` differs from the originals in
    /// `helpersCore` / `mixerHeadHeaderTail`; the 4-bit lane layout, the
    /// divisors, the `qdot` calls and the accumulation order are verbatim.
    static let tgHelpers = #"""

        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_tg(const threadgroup T* x, thread U* x_thread) {
          static_assert(bits == 4, "threadgroup vector load: 4-bit only");
          U sum = 0;
          for (int i = 0; i < values_per_thread; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_safe_tg(const threadgroup T* x, thread U* x_thread, int N) {
          static_assert(bits == 4, "threadgroup vector load: 4-bit only");
          U sum = 0;
          for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          for (int i = N; i < values_per_thread; i++) {
            x_thread[i] = 0;
          }
          return sum;
        }

        // `qmv_fast_reg` with the vector in threadgroup memory.
        template <typename T, int group_size, int bits, int results_per_simdgroup = 4>
        METAL_FUNC void qmv_fast_reg_tg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const threadgroup T* x,
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
            U sum = load_vector_tg<T, U, values_per_thread, bits>(x, x_thread);
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

        // `track_inject_qmv_row` with the vector in threadgroup memory.
        template <typename T, int group_size, int bits, int in_vec_size, int ROW, int UNR = 8>
        METAL_FUNC void track_inject_qmv_row_tg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const threadgroup T* x,
            device T* y,
            uint simd_lid) {
          constexpr int results_per_simdgroup = 1;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(ROW >= 0 && ROW < 4, "inject row");
          static_assert(in_vec_size > block_size, "K walk");

          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          thread U result[results_per_simdgroup] = {0};

          constexpr int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          constexpr int in_vec_size_g = in_vec_size / group_size;
          constexpr int out_row = ROW;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          y += out_row;

          constexpr int NFULL = (in_vec_size - 1) / block_size;
          constexpr int NR = 1;
          #pragma clang loop unroll_count(UNR)
          for (int i = 0; i < NFULL; i++) {
            U sum = load_vector_tg<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < NR; row++) {
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
          constexpr int k_end = NFULL * block_size;
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k_end - simd_lid * values_per_thread), 0, values_per_thread);
          if (remaining > 0) {
            U sum = load_vector_safe_tg<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < NR; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) {
              y[row] = static_cast<T>(result[row]);
            }
          }
        }
        """#

    static let header =
        TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + tgHelpers

    /// residual [1,1,H] (TILE) or [1,1,W], out [1,1,H], inject [1,1,HC],
    /// scale [W], down/inject weights  ->  stream [1,1,W], normed [1,1,W],
    /// lo [1,ND], act [1,ND], inj [1,HC].
    ///
    /// grid threads (32, NTILES * NSG, 1), threadgroup (32, NSG, 1).
    /// Tiles [0, ND / (NSG * RPS)) own down rows, RPS per simdgroup; the last
    /// tile (when the inject weight is present) owns the HC inject rows, one
    /// per simdgroup.
    ///
    /// One thread owns the same four consecutive lanes of EVERY hc stream for
    /// the whole kernel: it is virtual thread `32g + lane` of
    /// `track_inject_norm`'s 640-thread layout, for all HC of that layout's
    /// instances at once. That is what keeps the redundant pass cheap --
    /// `residual`, `scale` and the threadgroup store move four bf16 at a time,
    /// `out` is read once instead of once per stream, and the pre-norm row
    /// never makes a round trip through threadgroup memory: it stays in
    /// registers between the reduction and the scaling.
    static let source = """
        const int tile = (int)threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint tid = sg * 32 + lane;

        constexpr int N_READS = 4;
        constexpr uint NTHREADS = H / N_READS;
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        constexpr int NG = ((int)simd_groups + NSG - 1) / NSG;
        constexpr int NTD = ND / (NSG * RPS);
        constexpr int NTILES = NTD + (HAS_DINJ ? 1 : 0);
        constexpr int CHUNK = ((W / N_READS + NTILES - 1) / NTILES) * N_READS;
        static_assert(NSG >= HC, "at least one simdgroup per hc stream");
        static_assert(ND % (NSG * RPS) == 0, "down rows divide the tile");
        static_assert(HC == 4, "one-row inject tiles require four HC rows");
        static_assert(W == HC * H, "stream width");
        static_assert(H % (32 * N_READS) == 0, "the 640-thread layout is full");

        threadgroup T shx[W];
        threadgroup float local_sums[HC][32];
        threadgroup float inv_means[HC];

        for (uint i = tid; i < (uint)(HC * 32); i += (uint)(NSG * 32)) {
            local_sums[i / 32][i % 32] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- the norm's first pass, replayed. Lane `lane` of iteration `g` is
        // virtual thread 32g + lane; its four values per stream stay in
        // registers, and its four squares fold into `acc` in the reference's
        // order before the reference's `simd_sum`.
        T rr[NG][HC][N_READS];
        {
            T inj_t[HC];
            for (int h = 0; h < HC; ++h) { inj_t[h] = HAS_INJECT ? inject[h] : T(0); }
            for (int t = 0; t < NG; ++t) {
                const uint g = (uint)(sg + t * NSG);
                float acc[HC];
                for (int h = 0; h < HC; ++h) { acc[h] = 0.0f; }
                if (g < simd_groups) {
                    const uint d0 = (g * 32 + lane) * N_READS;
                    T ot[N_READS];
                    if (HAS_INJECT) {
                        for (int i = 0; i < N_READS; ++i) { ot[i] = out[d0 + i]; }
                    }
                    T rt[N_READS];
                    if (TILE) {
                        for (int i = 0; i < N_READS; ++i) { rt[i] = residual[d0 + i]; }
                    }
                    for (int h = 0; h < HC; ++h) {
                        const uint base = (uint)h * (uint)H;
                        for (int i = 0; i < N_READS; ++i) {
                            T r = TILE ? rt[i] : residual[base + d0 + i];
                            if (HAS_INJECT) {
                                T sp = ot[i] * inj_t[h];
                                r = r + sp;
                            }
                            rr[t][h][i] = r;
                            const float xf = static_cast<float>(r);
                            acc[h] += xf * xf;
                        }
                    }
                }
                for (int h = 0; h < HC; ++h) {
                    const float a = simd_sum(acc[h]);
                    if (lane == 0 && g < simd_groups) { local_sums[h][g] = a; }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg < (uint)HC) {
            const float total = simd_sum(local_sums[sg][lane]);
            if (lane == 0) {
                inv_means[sg] = metal::precise::rsqrt(total / (float)H + as_type<float>((uint)EPS_BITS));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- the norm's second pass. Every tile scales the whole row into its
        // own threadgroup copy; exactly one tile stores each element of
        // `stream` and `normed`, so no device write is duplicated.
        {
            const int start = tile * CHUNK;
            const int end = (start + CHUNK < W) ? (start + CHUNK) : W;
            for (int t = 0; t < NG; ++t) {
                const uint g = (uint)(sg + t * NSG);
                if (g < simd_groups) {
                    const uint d0 = (g * 32 + lane) * N_READS;
                    for (int h = 0; h < HC; ++h) {
                        const int gd = (int)((uint)h * (uint)H + d0);
                        const float im = inv_means[h];
                        const bool store = (gd >= start && gd < end);
                        for (int i = 0; i < N_READS; ++i) {
                            const T r = rr[t][h][i];
                            if (store) { stream[gd + i] = r; }
                            const T n = static_cast<T>(static_cast<float>(r) * im);
                            const T v = n * scale[gd + i];
                            shx[gd + i] = v;
                            if (store) { normed[gd + i] = v; }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- the mixer's down and inject walks over that row.
        if (tile < NTD) {
            float r[RPS];
            const int row0 = tile * (NSG * RPS) + (int)sg * RPS;
            qmv_fast_reg_tg<T, GS, BITS, RPS>(wd, sd, bd, shx, W, row0, lane, r);
            if (lane == 0) {
                for (int i = 0; i < RPS; ++i) {
                    const T l = static_cast<T>(r[i]);
                    lo[row0 + i] = l;
                    act[row0 + i] = mlx_silu(l);
                }
            }
        } else if (HAS_DINJ) {
            if (sg < (uint)HC) {
                switch ((int)sg) {
                    case 0: track_inject_qmv_row_tg<T, GS, BITS, W, 0>(wi, si, bi, shx, inj, lane); break;
                    case 1: track_inject_qmv_row_tg<T, GS, BITS, W, 1>(wi, si, bi, shx, inj, lane); break;
                    case 2: track_inject_qmv_row_tg<T, GS, BITS, W, 2>(wi, si, bi, shx, inj, lane); break;
                    case 3: track_inject_qmv_row_tg<T, GS, BITS, W, 3>(wi, si, bi, shx, inj, lane); break;
                }
            }
        }
        """

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_norm_down_inject",
        inputNames: ["residual", "out", "inject", "scale", "wd", "sd", "bd", "wi", "si", "bi"],
        outputNames: ["stream", "normed", "lo", "act", "inj"],
        source: source, header: header, ensureRowContiguous: true)

    /// True when this mixer's shapes and weights fit the fused launch.
    static func applies(
        residual: MLXArray, hcCount: Int, hidden: Int, down: TrackQuantWeight,
        inject: TrackQuantWeight?
    ) -> Bool {
        guard enabled, residual.ndim == 3, residual.dim(0) == 1, residual.dim(1) == 1 else {
            return false
        }
        guard hcCount == 4, hidden % 4 == 0, down.bits == 4, down.groupSize == 32,
            down.biases != nil, down.rows % (simdgroups * rowsPerSimdgroup) == 0,
            simdgroups >= hcCount,
            (hcCount * hidden) % 512 == 0
        else { return false }
        if let inject { return inject.bits == 4 && inject.groupSize == 32 && inject.biases != nil && inject.rows == hcCount }
        return true
    }

    static func normDownInject(
        residual: MLXArray, out: MLXArray?, inject: MLXArray?, scale: MLXArray,
        hcCount: Int, hidden: Int, eps: Float, tile: Bool,
        down: TrackQuantWeight, injectW: TrackQuantWeight?
    ) -> (stream: MLXArray, normed: MLXArray, lo: MLXArray, act: MLXArray, inj: MLXArray) {
        let W = hcCount * hidden
        let ND = down.rows
        let hasInject = out != nil
        let iw = injectW ?? down
        let tiles = ND / (simdgroups * rowsPerSimdgroup) + (injectW != nil ? 1 : 0)
        let outs = kernel(
            [
                residual, out ?? residual, inject ?? residual, scale,
                down.weight, down.scales, down.biases!, iw.weight, iw.scales, iw.biases!,
            ],
            template: [
                ("T", residual.dtype), ("GS", down.groupSize), ("BITS", down.bits),
                ("H", hidden), ("HC", hcCount), ("W", W), ("ND", ND),
                ("NSG", simdgroups), ("RPS", rowsPerSimdgroup),
                ("EPS_BITS", Int(eps.bitPattern)), ("HAS_INJECT", hasInject), ("TILE", tile),
                ("HAS_DINJ", injectW != nil),
            ],
            grid: (32, tiles * simdgroups, 1), threadGroup: (32, simdgroups, 1),
            outputShapes: [[1, 1, W], [1, 1, W], [1, ND], [1, ND], [1, hcCount]],
            outputDTypes: [
                residual.dtype, residual.dtype, residual.dtype, residual.dtype, residual.dtype,
            ])
        return (outs[0], outs[1], outs[2], outs[3], outs[4])
    }
}
