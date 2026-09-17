// TrackPleFusedKV.swift -- fused key|value projection for the PLE block.
//
// The PLE block applies its key and value projections to the same embedded
// rows back to back. When both are compatible quantized weights, one
// quantizedMM over the row-concatenated weight produces both: each output
// column is an independent dot over the same K, so column j of the fused
// result is exactly what the separate calls produced for their row j. This
// is the same fusion TrackMultiProj already applies to the shared expert's
// gate|up pair.
import Foundation
import MLX
import MLXLLM

enum TrackPleFusedKV {
    /// Fused [key rows | value rows] weight per PLE embedding, built once.
    nonisolated(unsafe) private static var fused: [ObjectIdentifier: TrackQuantWeight] = [:]
    /// Embeddings whose projections cannot fuse (dense or incompatible).
    nonisolated(unsafe) private static var unfusable: Set<ObjectIdentifier> = []

    /// The concatenated key|value projection, or nil when the two
    /// projections are not compatible quantized weights — callers then keep
    /// the separate GEMVs. In the fused output, columns [0, keyRows) are the
    /// key projection and [keyRows, keyRows + valueRows) the value.
    static func keyValue(_ p: TrackPLE) -> TrackQuantWeight? {
        let id = ObjectIdentifier(p.embedding)
        if let f = fused[id] { return f }
        if unfusable.contains(id) { return nil }
        guard case .quant(let kq) = p.keyProj, case .quant(let vq) = p.valueProj,
            kq.compatible(vq)
        else {
            unfusable.insert(id)
            return nil
        }
        let f = TrackQuantWeight.concat([kq, vq])
        eval(f.weight, f.scales, f.biases ?? f.weight)
        fused[id] = f
        return f
    }
}
