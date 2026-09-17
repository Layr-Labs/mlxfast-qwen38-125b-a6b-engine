import Foundation
import MLX

enum TrackPrefillRouter {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_ROUTER_BF16_STORAGE"] != "0"
        && (Int(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "1") ?? 0) != 0

    private static let supportsNAX: Bool = {
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return false
        }
        let arch = GPU.deviceInfo().architecture
        guard let generation = Int(arch.dropLast().suffix(2)), let family = arch.last else {
            return false
        }
        return generation >= (family == "p" ? 18 : 17)
    }()

    private static let partialKernel = MLXFast.metalKernel(
        name: "track_router_bf16_storage",
        inputNames: ["x", "w"], outputNames: ["partials"], source: source,
        header: TrackPrefillIndirect.metalHeader + loopHeader, ensureRowContiguous: true)

    private static let sumKernel = MLXFast.metalKernel(
        name: "track_router_split_sum", inputNames: ["partials"], outputNames: ["y"],
        source: sumSource, ensureRowContiguous: true)

    static func apply(x: MLXArray, w: MLXArray) -> MLXArray? {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 32, x.dim(1) <= 1024,
            x.dim(2) == 2560, x.dtype == .bfloat16,
            w.shape == [512, 2560], w.dtype == .bfloat16
        else { return nil }
        let rows = x.dim(1), tilesM = (rows + 63) / 64
        let swizzle = tilesM <= 3 ? 1 : 2
        let groups = 8 * swizzle * ((tilesM + swizzle - 1) / swizzle) * 2
        let partials = partialKernel(
            [x, w], template: [("M", rows)],
            grid: (groups * 32, 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[2, rows, 512]], outputDTypes: [.float32])[0]
        return sumKernel(
            [partials], template: [("M", rows)],
            grid: (512, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, 512]], outputDTypes: [.float32])[0]
    }

    /// Dedicated no-gate prefill route: the split-K partial sum folded into
    /// the top-K + softmax walk, one simdgroup per row, four rows per
    /// threadgroup. `logits` never materializes — the walk reads
    /// `partials[row] + partials[R + row]` in the same order `sumKernel`
    /// adds them, so the selected values are bit-identical. No gate GEMV,
    /// no threadgroup arrays, no barrier: the simd_max/simd_min pair already
    /// broadcasts each winner across the walk's own simdgroup.
    private static let fusedRouteKernel = MLXFast.metalKernel(
        name: "track_router_prefill_route",
        inputNames: ["partials"], outputNames: ["idx", "w"],
        source: fusedRouteSource, header: TrackFastKernels.exactHeader,
        ensureRowContiguous: true)

    /// partialKernel + fusedRouteKernel: `(idx [1,S,K] uint32, w [1,S,K] f32)`,
    /// or nil outside the supported window. Same gates as `apply`.
    static func applyRouted(x: MLXArray, w weight: MLXArray, topK: Int)
        -> (idx: MLXArray, weights: MLXArray)?
    {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 32, x.dim(1) <= 1024,
            x.dim(2) == 2560, x.dtype == .bfloat16,
            weight.shape == [512, 2560], weight.dtype == .bfloat16,
            topK >= 1 && topK <= 32
        else { return nil }
        let rows = x.dim(1), tilesM = (rows + 63) / 64
        let swizzle = tilesM <= 3 ? 1 : 2
        let groups = 8 * swizzle * ((tilesM + swizzle - 1) / swizzle) * 2
        let partials = partialKernel(
            [x, weight], template: [("M", rows)],
            grid: (groups * 32, 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[2, rows, 512]], outputDTypes: [.float32])[0]
        let outs = fusedRouteKernel(
            [partials], template: [("E", 512), ("K", topK), ("ROWS", rows)],
            grid: (32, ((rows + 3) / 4) * 4, 1), threadGroup: (32, 4, 1),
            outputShapes: [[rows, topK], [rows, topK]],
            outputDTypes: [.uint32, .float32])
        return (outs[0].reshaped(1, rows, topK), outs[1].reshaped(1, rows, topK))
    }

    static let fusedRouteSource = """
        constexpr int E_PER = (E + 31) / 32;
        const uint row = threadgroup_position_in_grid.y * 4
            + simdgroup_index_in_threadgroup;
        if (row >= (uint)ROWS) { return; }
        const uint lane = thread_index_in_simdgroup;
        constexpr int N_READS = 4;
        float ld[N_READS];
        uint selected[N_READS];
        for (int i = 0; i < N_READS; ++i) {
            ld[i] = -INFINITY;
            selected[i] = 0xffffffffu;
        }
        const device float* pr = partials + (size_t)row * (size_t)E;
        // Each lane owns E_PER experts: e = lane + 32 * j (strided so a tie at
        // the same value resolves to the lowest index across lanes too). The
        // two split-K halves are added in sumKernel's order.
        float v[E_PER];
        bool taken[E_PER];
        for (int j = 0; j < E_PER; ++j) {
            const int e = (int)lane + 32 * j;
            v[j] = (e < E) ? (pr[e] + pr[(size_t)ROWS * (size_t)E + e]) : -INFINITY;
            taken[j] = (e >= E);
        }
        for (int k = 0; k < K; ++k) {
            // lane-local best: largest value, then lowest index
            float bv = -INFINITY; int bj = -1;
            for (int j = 0; j < E_PER; ++j) {
                if (!taken[j] && (v[j] > bv)) { bv = v[j]; bj = j; }
            }
            const float gmax = simd_max(bv);
            const uint cand = (bv == gmax && bj >= 0)
                ? (uint)(lane + 32 * bj) : 0xffffffffu;
            const uint gidx = simd_min(cand);
            for (int i = 0; i < N_READS; ++i) {
                if (k == (int)lane * N_READS + i) { ld[i] = gmax; selected[i] = gidx; }
            }
            if (gidx == (uint)(lane + 32 * bj) && bj >= 0) { taken[bj] = true; }
        }
        // softmax_single_row over the K selected logits (AccT = float)
        float maxval = -FLT_MAX;
        for (int i = 0; i < N_READS; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
        maxval = simd_max(maxval);
        float normalizer = 0;
        for (int i = 0; i < N_READS; i++) {
            float exp_x = fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        normalizer = 1 / normalizer;
        for (int i = 0; i < N_READS; i++) {
            const int p = (int)lane * N_READS + i;
            if (p < K) {
                w[(size_t)row * K + p] = ld[i] * normalizer;
                idx[(size_t)row * K + p] = selected[i];
            }
        }
        """

    static let source = #"""
        constexpr int tiles_m = (M + 63) / 64;
        constexpr int swizzle_log = tiles_m <= 3 ? 0 : 1;
        constexpr int tn_swizzled = 8 << swizzle_log;
        constexpr int tm_swizzled = (tiles_m + (1 << swizzle_log) - 1) >> swizzle_log;
        constexpr int tiles_per_partition = tn_swizzled * tm_swizzled;
        const int linear_tid = threadgroup_position_in_grid.x;
        const int partition = linear_tid / tiles_per_partition;
        const int xy_flat = linear_tid % tiles_per_partition;
        const int grid_x = xy_flat % tn_swizzled;
        const int grid_y = xy_flat / tn_swizzled;
        const int tid_y = (grid_y << swizzle_log) + (grid_x & ((1 << swizzle_log) - 1));
        const int tid_x = grid_x >> swizzle_log;
        if (tid_y >= tiles_m) { return; }
        const int c_row = tid_y * 64;
        const int c_col = tid_x * 64;
        const int k_start = partition * 2048;
        const int partition_k = min(2048, 2560 - k_start);
        const short tm = 32 * (simdgroup_index_in_threadgroup / 2);
        const short tn = 32 * (simdgroup_index_in_threadgroup % 2);
        const short sm = (M % 64 == 0) ? 32 : short(min(32, M - c_row - tm));
        const device bfloat16_t* A = x + size_t(c_row + tm) * 2560 + k_start;
        const device bfloat16_t* B = w + size_t(c_col + tn) * 2560 + k_start;
        device float* C = partials + size_t(partition) * M * 512 + size_t(c_row + tm) * 512 + c_col + tn;
        NAXTile<float, 2, 2> Dtile;
        dispatch_bool(M % 64 == 0 || sm == 32, [&](auto aligned_m) {
            Dtile = track_router_loop<bfloat16_t, 32, 32, 32, 256, false, true,
                aligned_m.value, true, true, float>(
                A, B, 2560, 2560, partition_k, partition_k / 256, sm, 32);
        });
        dispatch_bool(M % 64 == 0 || sm == 32, [&](auto aligned_m) {
            if constexpr (aligned_m) { Dtile.store(C, 512); }
            else { Dtile.store_safe(C, 512, short2(32, sm)); }
        });
        """#

    static let sumSource = """
        const uint i = thread_position_in_grid.y * 512 + thread_position_in_grid.x;
        float total = 0.0f;
        total += partials[i];
        total += partials[M * 512 + i];
        y[i] = total;
        """
}
