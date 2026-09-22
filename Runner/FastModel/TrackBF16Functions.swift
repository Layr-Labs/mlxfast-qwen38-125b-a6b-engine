import MLX

enum TrackBF16Functions {
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

    private static let siluKernel = MLXFast.metalKernel(
        name: "track_bf16_silu_table",
        inputNames: [], outputNames: ["table"],
        source: siluSource, header: TrackFastKernels.exactHeader)

    /// `mlx_silu` evaluated on every bfloat16 bit pattern, so a lookup returns
    /// the same bits the inline call returns for a bfloat16 argument.
    nonisolated(unsafe) static let silu: MLXArray = {
        let table = siluKernel(
            [], grid: (65536, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[65536]], outputDTypes: [.bfloat16], stream: .gpu)[0]
        eval(table)
        return table
    }()

    static let siluSource = #"""
        const uint i = thread_position_in_grid.x;
        table[i] = mlx_silu(as_type<bfloat16_t>(ushort(i)));
        """#
}
