import MLX

enum TrackBF16Functions {
    // GDN's output gate evaluates sigmoid in FP32 after widening a BF16
    // input. Keep a separate full-domain FP32 table; the existing BF16
    // sigmoid table has different intermediate rounding and cannot be reused.
    private static let sigmoidFloatKernel = MLXFast.metalKernel(
        name: "track_bf16_input_float_sigmoid_table",
        inputNames: [], outputNames: ["table"],
        source: sigmoidFloatSource, header: TrackFastKernels.exactHeader)

    nonisolated(unsafe) static let sigmoidFloat: MLXArray = {
        let table = sigmoidFloatKernel(
            [], grid: (65536, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[65536]], outputDTypes: [.float32], stream: .gpu)[0]
        eval(table)
        return table
    }()

    static let sigmoidFloatSource = #"""
        const uint i = thread_position_in_grid.x;
        const auto z = as_type<bfloat16_t>(ushort(i));
        table[i] = mlx_sigmoid(static_cast<float>(z));
        """#

    private static let sigmoidKernel = MLXFast.metalKernel(
        name: "track_bf16_sigmoid_table",
        inputNames: [], outputNames: ["table"],
        source: sigmoidSource, header: TrackFastKernels.exactHeader)

    nonisolated(unsafe) static let sigmoid: MLXArray = {
        let table = sigmoidKernel(
            [], grid: (65536, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[65536]], outputDTypes: [.bfloat16], stream: .gpu)[0]
        eval(table)
        return table
    }()

    static let sigmoidSource = #"""
        const uint i = thread_position_in_grid.x;
        table[i] = mlx_sigmoid(as_type<bfloat16_t>(ushort(i)));
        """#
}
