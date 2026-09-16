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

    /// Fold the two K partitions into each lane's routing registers. This
    /// removes the intermediate 512-logit write/read and the sum-only launch.
    /// The addition is deliberately 0 + partition0 + partition1, as sumSource.
    private static let routeKernel = MLXFast.metalKernel(
        name: "track_prefill_router_sum_top10_v3",
        inputNames: ["partials", "x", "wg", "sgw", "bgw"],
        outputNames: ["idx", "w", "gate"],
        source: TrackFastMoEKernels.routeSource
            .replacingOccurrences(
                of: "const device float* lr = logits + (size_t)row * (size_t)E;",
                with: "const device float* lr = partials + (size_t)row * (size_t)E;")
            .replacingOccurrences(
                of: "v[j] = (e < E) ? lr[e] : -INFINITY;",
                with: "float total = 0.0f; total += lr[e]; total += lr[M * E + e]; v[j] = total;"),
        header: TrackFastKernels.mixerHeadHeader + TrackFastMoEKernels.wideDecls,
        ensureRowContiguous: true)

    static func routed(x: MLXArray, w: MLXArray, topK: Int)
        -> (idx: MLXArray, w: MLXArray)?
    {
        guard enabled, supportsNAX, StreamOrDevice.default.stream === Stream.gpu,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 256, x.dim(1) <= 1024,
            x.dim(2) == 2560, x.dtype == .bfloat16,
            w.shape == [512, 2560], w.dtype == .bfloat16, topK == 10
        else { return nil }
        let rows = x.dim(1), tilesM = (rows + 63) / 64
        let swizzle = tilesM <= 3 ? 1 : 2
        let groups = 8 * swizzle * ((tilesM + swizzle - 1) / swizzle) * 2
        let partials = partialKernel(
            [x, w], template: [("M", rows)],
            grid: (groups * 32, 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[2, rows, 512]], outputDTypes: [.float32])[0]
        let out = routeKernel(
            [partials, x, x, x, x],
            template: [("M", rows), ("E", 512), ("K", topK), ("T", x.dtype),
                       ("GS", 32), ("BITS", 4), ("KD", 2560), ("VPT", rows), ("HAS_GATE", false)],
            grid: (32, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[1, rows, topK], [1, rows, topK], [1]],
            outputDTypes: [.uint32, .float32, x.dtype])
        return (out[0], out[1])
    }

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
