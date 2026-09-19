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

    private static let fusedRouteKernel = MLXFast.metalKernel(
        name: "track_router_split_route",
        inputNames: ["partials", "x", "wg", "sgw", "bgw"],
        outputNames: ["idx", "w", "gate"], source: fusedRouteSource,
        header: TrackFastKernels.mixerHeadHeader + TrackFastMoEKernels.wideHelpers,
        ensureRowContiguous: true)

    private static func makePartials(x: MLXArray, w: MLXArray)
        -> (partials: MLXArray, rows: Int)?
    {
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
        return (partials, rows)
    }

    static func apply(x: MLXArray, w: MLXArray) -> MLXArray? {
        guard let (partials, rows) = makePartials(x: x, w: w) else { return nil }
        return sumKernel(
            [partials], template: [("M", rows)],
            grid: (512, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, 512]], outputDTypes: [.float32])[0]
    }

    /// Wide prefill route directly from the two f32 router partitions. The
    /// partition GEMM and the old `apply` fallback stay unchanged.
    static func applyRouted(x: MLXArray, w: MLXArray)
        -> (idx: MLXArray, w: MLXArray)?
    {
        guard let (partials, rows) = makePartials(x: x, w: w) else { return nil }
        let routed = fusedRouteKernel(
            [partials, x, x, x, x],
            template: [
                ("E", 512), ("K", 10), ("T", x.dtype), ("GS", 32),
                ("BITS", 4), ("KD", 2560), ("VPT", rows), ("HAS_GATE", false),
            ],
            grid: (32, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[1, rows, 10], [1, rows, 10], [1, rows]],
            outputDTypes: [.uint32, .float32, x.dtype])
        return (routed[0], routed[1])
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

    private static func routeSourceVariant(_ source: String) -> String {
        let replacements = [
            (
                "const device float* lr = logits + (size_t)row * (size_t)E;",
                "const device float* p0 = partials + (size_t)row * 512;\n"
                    + "const device float* p1 = partials + (size_t)VPT * 512 + (size_t)row * 512;"
            ),
            (
                "v[j] = (e < E) ? lr[e] : -INFINITY;",
                "if (e < E) {\n"
                    + "    float total = 0.0f;\n"
                    + "    total += p0[e];\n"
                    + "    total += p1[e];\n"
                    + "    v[j] = total;\n"
                    + "} else {\n"
                    + "    v[j] = -INFINITY;\n"
                    + "}"
            ),
        ]
        var result = source
        for (old, new) in replacements {
            precondition(
                result.components(separatedBy: old).count == 2,
                "TrackPrefillRouter: route source anchor is not unique")
            result = result.replacingOccurrences(of: old, with: new)
        }
        return result
    }


    // Derive the fused route from the production route tail. The two replacements
    // add the original zero + partition-0 + partition-1 sum before the unchanged
    // stable top-k and float32 softmax code.
    private static let fusedRouteSource = routeSourceVariant(TrackFastMoEKernels.routeSource)
}
