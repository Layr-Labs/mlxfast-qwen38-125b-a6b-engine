import Foundation
import MLX

enum TrackPrefillShared {
    private static func weights(_ m: TrackMoE)
        -> (gateUp: TrackQuantWeight, down: TrackQuantWeight)?
    {
        guard case .quant(let gu)? = m.sharedGateUp.fused,
            case .quant(let down) = m.sharedDown,
            gu.mode == .affine, down.mode == .affine,
            gu.bits == 4, down.bits == 4, gu.groupSize == 32, down.groupSize == 32,
            gu.weight.shape == [1280, 320], gu.scales.shape == [1280, 80],
            gu.biases?.shape == gu.scales.shape,
            down.weight.shape == [2560, 80], down.scales.shape == [2560, 20],
            down.biases?.shape == down.scales.shape,
            gu.weight.dtype == .uint32, down.weight.dtype == .uint32,
            gu.scales.dtype == .bfloat16, gu.biases?.dtype == .bfloat16,
            down.scales.dtype == .bfloat16, down.biases?.dtype == .bfloat16
        else { return nil }
        return (gu, down)
    }

    static func supports(_ m: TrackMoE, x: MLXArray) -> Bool {
        let d = m.expertDown
        return x.ndim == 3 && x.dim(0) == 1 && x.dim(1) >= 256 && x.dim(1) < 3277
            && x.dim(2) == 2560 && x.dtype == .bfloat16
            && m.topK == 10 && m.sharedHidden == 640 && weights(m) != nil
            && d.w.shape == [512, 2560, 80] && d.s.shape == [512, 2560, 20]
            && d.b.shape == d.s.shape && d.w.dtype == .uint32
            && d.s.dtype == .bfloat16 && d.b.dtype == .bfloat16
    }

    private static let metadataLock = NSLock()
    nonisolated(unsafe) private static var metadataCache: [Int: MLXArray] = [:]

    private static func metadata(_ rows: Int) -> MLXArray {
        metadataLock.lock(); defer { metadataLock.unlock() }
        if let cached = metadataCache[rows] { return cached }
        var values = [UInt32](repeating: 0, count: rows)
        values.append(contentsOf: (0..<rows).map(UInt32.init))
        for start in stride(from: 0, to: rows, by: 32) {
            values.append(UInt32(start))
            values.append(UInt32(min(rows, start + 32)))
        }
        let result = MLXArray(values)
        eval(result)
        if metadataCache.count >= 8 { metadataCache.removeAll(keepingCapacity: true) }
        metadataCache[rows] = result
        return result
    }

    static func gateUp(
        _ m: TrackMoE, x: MLXArray, sortedIDs: MLXArray, tokenRows: MLXArray,
        tiles: MLXArray, maxTiles: Int
    ) -> MLXArray? {
        guard let shared = weights(m) else { return nil }
        let g = m.expertGate, u = m.expertUp, q = shared.gateUp
        let s = x.dim(1), r = sortedIDs.size
        return gateUpKernel(
            [x, g.w, g.s, g.b, u.w, u.s, u.b, sortedIDs, tokenRows, tiles,
             q.weight, q.scales, q.biases!, metadata(s)],
            template: [("T", x.dtype), ("R", r), ("S", s), ("RT", maxTiles)],
            grid: (320, (maxTiles + (s + 31) / 32) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[r + s, 1, 640]], outputDTypes: [.bfloat16])[0]
    }

    static func down(
        _ m: TrackMoE, activated: MLXArray, sortedIDs: MLXArray, tokenRows: MLXArray,
        tiles: MLXArray, maxTiles: Int, sharedRows s: Int
    ) -> MLXArray? {
        guard s >= 256, s < 3277, let shared = weights(m) else { return nil }
        let d = m.expertDown, q = shared.down, r = sortedIDs.size
        return downKernel(
            [activated, d.w, d.s, d.b, sortedIDs, tokenRows, tiles,
             q.weight, q.scales, q.biases!, metadata(s)],
            template: [("T", activated.dtype), ("R", r), ("S", s), ("RT", maxTiles)],
            grid: (640, (maxTiles + (s + 31) / 32) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[r + s, 1, 2560]], outputDTypes: [.bfloat16])[0]
    }

    private static let gateUpKernel = MLXFast.metalKernel(
        name: "track_prefill_shared_gate_up",
        inputNames: ["x", "w0", "scales0", "biases0", "w1", "scales1", "biases1",
                     "indices", "token_rows", "tiles", "sw", "ss", "sb", "meta"],
        outputNames: ["y"], source: gateUpSource,
        header: TrackPrefillIndirect.metalHeader, ensureRowContiguous: true)

    private static let downKernel = MLXFast.metalKernel(
        name: "track_prefill_shared_down",
        inputNames: ["x", "w", "scales", "biases", "indices", "token_rows", "tiles",
                     "sw", "ss", "sb", "meta"],
        outputNames: ["y"], source: downSource,
        header: TrackPrefillIndirect.metalHeader, ensureRowContiguous: true)

    static let gateUpSource = #"""
        constexpr int N = 640, K = 2560;
        uint3 tid = threadgroup_position_in_grid;
        const bool shared = tid.y >= RT;
        if (shared) { tid.y -= RT; }
        alignas(16) threadgroup T Ws0[64 * 72];
        alignas(16) threadgroup T Ws1[64 * 72];
        alignas(16) threadgroup T As[32 * 72];
        device T* out = y + (shared ? size_t(R) * N : 0);
        track_prefill_indirect_gu<T, 32, 4, 32, 64, 64, 2, 2, true, true, N>(
            x, shared ? sw : w0, shared ? ss : scales0, shared ? sb : biases0,
            shared ? sw + N * (K / 8) : w1,
            shared ? ss + N * (K / 32) : scales1,
            shared ? sb + N * (K / 32) : biases1,
            shared ? meta : indices, shared ? meta + S : token_rows,
            shared ? meta + 2 * S : tiles, out, out, N, K, Ws0, Ws1, As, tid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

    static let downSource = #"""
        constexpr int N = 2560, K = 640;
        uint3 tid = threadgroup_position_in_grid;
        const bool shared = tid.y >= RT;
        if (shared) { tid.y -= RT; }
        alignas(16) threadgroup T Ws[128 * 40];
        alignas(16) threadgroup T As[32 * 40];
        track_prefill_indirect<T, 32, 4, 32, 128, 32, 2, 2, true, N>(
            x + (shared ? size_t(R) * K : 0),
            shared ? sw : w, shared ? ss : scales, shared ? sb : biases,
            shared ? meta : indices, shared ? meta + S : token_rows,
            shared ? meta + 2 * S : tiles, y + (shared ? size_t(R) * N : 0),
            N, K, Ws, As, tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
