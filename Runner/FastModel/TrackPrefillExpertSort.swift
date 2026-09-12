import MLX
import MLXLMCommon

enum TrackPrefillExpertSort {
    nonisolated(unsafe) private static let inverseKernel = MLXFast.metalKernel(
        name: "track_prefill_inverse_permutation",
        inputNames: ["order"], outputNames: ["inverse"],
        source: """
            const uint i = thread_position_in_grid.x;
            if (i < N) inverse[order[i]] = i;
            """,
        ensureRowContiguous: true)

    static func sortedInputs(x: MLXArray, indices: MLXArray)
        -> (MLXArray, MLXArray, MLXArray)
    {
        guard indices.size > 0, indices.size <= Int(Int32.max),
            StreamOrDevice.default.stream === Stream.gpu
        else { return gatherSort(x: x, indices: indices) }
        let topK = indices.dim(-1)
        let flat = indices.flattened()
        let order = argSort(flat)
        let inverse = inverseKernel(
            [order], template: [("N", order.size)],
            grid: (order.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[order.size]], outputDTypes: [.uint32])[0]
        return (
            x.flattened(start: 0, end: -3)[order.floorDivide(topK)],
            flat[order], inverse
        )
    }
}
