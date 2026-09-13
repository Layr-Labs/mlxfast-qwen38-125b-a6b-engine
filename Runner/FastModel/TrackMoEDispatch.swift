// TrackMoEDispatch.swift -- expert-segment gather offsets from fused top-k ids.
//
// Prefill MoE apply used `gatherSort` / `argSort` (stable) to put (token, slot)
// pairs in expert-major, token-index-minor order so `gather_qmm` can walk
// contiguous segments. That permutation is a histogram of the K ids per token,
// an exclusive prefix-sum, and a scatter -- not a property of the grouped
// GEMV itself (the GEMV binary-searches the already-sorted ids).
//
// Same-launch as top-k is infeasible: `track_moe_route` is one threadgroup per
// token and Metal has no grid-wide barrier, so a 512-bin histogram cannot be
// scanned until every row has written its K ids. A second kernel on the same
// stream, with no host eval, builds the CSR offsets and the gather permutation.
// Order is deterministic: expert id, then original flattened slot (token, then
// top-k slot) -- byte-identical to a stable argsort of the flattened ids.
//
// `TRACK_ROUTER_DISPATCH_FUSE=0` restores gatherSort. Unexpected dtypes,
// expert counts, or slot counts also fall back.

import Foundation
import MLX

enum TrackMoEDispatch {
    /// Kill switch. Default ON; `=0` keeps today's argsort construction.
    static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["TRACK_ROUTER_DISPATCH_FUSE"] != "0"
    }

    struct Pack {
        let sortedIDs: MLXArray
        let tokenRows: MLXArray
        let inverse: MLXArray
        let offsets: MLXArray
    }

    struct CPUPack: Equatable {
        let sortedIDs: [UInt32]
        let tokenRows: [UInt32]
        let inverse: [UInt32]
        let offsets: [UInt32]
    }

    /// Histogram + exclusive prefix-sum + token-major scatter. `indices` is
    /// token-major flattened top-k expert ids, each in `[0, experts)`.
    static func cpuPack(indices: [UInt32], tokens: Int, topK: Int, experts: Int) -> CPUPack {
        precondition(tokens >= 0 && topK >= 1 && experts >= 1)
        precondition(indices.count == tokens * topK)
        var hist = [UInt32](repeating: 0, count: experts)
        for e in indices {
            precondition(Int(e) < experts)
            hist[Int(e)] += 1
        }
        var offsets = [UInt32](repeating: 0, count: experts + 1)
        for e in 0..<experts {
            offsets[e + 1] = offsets[e] + hist[e]
        }
        var cursor = Array(offsets[0..<experts])
        var sortedIDs = [UInt32](repeating: 0, count: indices.count)
        var tokenRows = [UInt32](repeating: 0, count: indices.count)
        var inverse = [UInt32](repeating: 0, count: indices.count)
        for i in 0..<indices.count {
            let e = Int(indices[i])
            let pos = Int(cursor[e])
            cursor[e] += 1
            sortedIDs[pos] = indices[i]
            tokenRows[pos] = UInt32(i / topK)
            inverse[i] = UInt32(pos)
        }
        return CPUPack(
            sortedIDs: sortedIDs, tokenRows: tokenRows, inverse: inverse, offsets: offsets)
    }

    static func shouldFuse(
        enabled: Bool, onGPU: Bool, dtype: DType, experts: Int, slots: Int, topK: Int
    ) -> Bool {
        enabled && onGPU && dtype == .uint32 && experts >= 1 && experts <= 512
            && topK >= 1 && slots >= 64 && slots % topK == 0
    }

    /// Device pack, or `nil` to keep gatherSort. No host eval.
    static func pack(indices: MLXArray, experts: Int) -> Pack? {
        let topK = indices.dim(-1)
        let slots = indices.size
        let onGPU = StreamOrDevice.default.stream === Stream.gpu
        guard
            shouldFuse(
                enabled: isEnabled, onGPU: onGPU, dtype: indices.dtype, experts: experts,
                slots: slots, topK: topK)
        else { return nil }
        let flat = indices.reshaped(slots)
        let tg = ((experts + 31) / 32) * 32
        let outs = kernel(
            [flat],
            template: [("E", experts), ("K", topK)],
            grid: (tg, 1, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[slots], [slots], [slots], [experts + 1]],
            outputDTypes: [.uint32, .uint32, .uint32, .uint32])
        return Pack(sortedIDs: outs[0], tokenRows: outs[1], inverse: outs[2], offsets: outs[3])
    }

    // One threadgroup: thread 0 fills the E-bin histogram and scatters in
    // original-slot order (stable); every lane runs the exclusive scan.
    static let kernelSource = """
        const uint lid = thread_index_in_threadgroup;
        const uint nslots = (uint)idx_shape[0];
        threadgroup uint hist[E];
        threadgroup uint offs[E + 1];
        if (lid < (uint)E) { hist[lid] = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            for (uint i = 0; i < nslots; ++i) { hist[idx[i]] += 1u; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < (uint)E; stride <<= 1) {
            const uint addend = (lid < (uint)E && lid >= stride) ? hist[lid - stride] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < (uint)E && lid >= stride) { hist[lid] += addend; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid < (uint)E) { offs[lid] = (lid == 0) ? 0u : hist[lid - 1]; }
        if (lid == 0) { offs[E] = (E > 0) ? hist[E - 1] : 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < (uint)E) { offsets[lid] = offs[lid]; }
        if (lid == 0) { offsets[E] = offs[E]; }
        if (lid < (uint)E) { hist[lid] = offs[lid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            for (uint i = 0; i < nslots; ++i) {
                const uint e = idx[i];
                const uint pos = hist[e];
                hist[e] = pos + 1u;
                sorted_ids[pos] = e;
                token_rows[pos] = i / (uint)K;
                inverse[i] = pos;
            }
        }
        """

    static let kernel = MLXFast.metalKernel(
        name: "track_moe_dispatch",
        inputNames: ["idx"],
        outputNames: ["sorted_ids", "token_rows", "inverse", "offsets"],
        source: kernelSource, ensureRowContiguous: true)
}
