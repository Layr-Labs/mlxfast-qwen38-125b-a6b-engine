import Foundation
import MLX
import MLXLMCommon

enum TrackPrefillIndirect {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_PREFILL_INDIRECT_ACTIVATIONS"] != "0"

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

    private static let kernel = MLXFast.metalKernel(
        name: "track_prefill_indirect_activations",
        inputNames: ["x", "w", "scales", "biases", "indices", "token_rows"],
        outputNames: ["y"], source: source, header: metalHeader,
        ensureRowContiguous: true)

    static func apply(_ m: TrackMoE, x: MLXArray, indices: MLXArray)
        -> (activated: MLXArray, sortedIDs: MLXArray, inverse: MLXArray)?
    {
        guard enabled, supportsNAX, StreamOrDevice.default.stream == Stream.gpu,
            m.expertBits == 4, m.expertGroupSize == 32,
            x.ndim == 3, x.dim(0) == 1, x.dim(2) == 2560, x.dtype == .bfloat16,
            indices.ndim == 3, indices.dim(0) == 1, indices.dim(1) == x.dim(1),
            indices.dim(2) > 0, indices.dtype == .uint32,
            indices.size >= 2048, indices.size < 512 * 64
        else { return nil }
        let g = m.expertGate
        let u = m.expertUp
        guard g.w.shape == [512, 640, 320], u.w.shape == g.w.shape,
            g.s.shape == [512, 640, 80], g.b.shape == g.s.shape,
            u.s.shape == g.s.shape, u.b.shape == g.s.shape,
            g.w.dtype == .uint32, u.w.dtype == .uint32,
            g.s.dtype == .bfloat16, g.b.dtype == .bfloat16,
            u.s.dtype == .bfloat16, u.b.dtype == .bfloat16
        else { return nil }

        let flatIDs = indices.flattened()
        let sortedIDs: MLXArray
        let inverse: MLXArray
        let tokenRows: MLXArray
        if let c = TrackPrefillSort.apply(
            flatIDs: flatIDs, experts: g.w.dim(0), topK: indices.dim(2))
        {
            // The identical permutation in two launches (see TrackPrefillSort).
            (sortedIDs, tokenRows, inverse) = (c.sortedIDs, c.tokenRows, c.inverse)
        } else {
            let order = argSort(flatIDs)
            inverse = argSort(order)
            sortedIDs = flatIDs[order]
            tokenRows = order.floorDivide(indices.dim(2))
        }
        let rows = indices.size
        func project(_ bank: (w: MLXArray, s: MLXArray, b: MLXArray)) -> MLXArray {
            kernel(
                [x, bank.w, bank.s, bank.b, sortedIDs, tokenRows],
                template: [("T", x.dtype), ("M", rows), ("N", 640), ("K", 2560)],
                grid: (10 * 32, ((rows + 31) / 32) * 2, 2),
                threadGroup: (32, 2, 2),
                outputShapes: [[rows, 1, 640]], outputDTypes: [.bfloat16])[0]
        }
        let up = project(u)
        let gate = project(g)
        return (compiledSiluProduct(gate, up), sortedIDs, inverse)
    }

    static let source = #"""
        threadgroup T Ws[64 * 72];
        threadgroup T As[32 * 72];
        track_prefill_indirect<T, 32, 4, 32, 64, 64, 2, 2, true>(
            x, w, scales, biases, indices, token_rows, y,
            M, N, K, Ws, As, threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
