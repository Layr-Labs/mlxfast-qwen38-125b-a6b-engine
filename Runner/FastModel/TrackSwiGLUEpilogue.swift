// TrackSwiGLUEpilogue.swift -- fuse `silu(gate) * up` into the expert gate|up
// GEMV. The fused Metal kernel keeps both halves in registers, applies
// `mlx_silu` in the activation dtype, and writes only the product.
//
// Bit-exact with the two-kernel path (`track_moe_gate_up` then `track_swiglu2`):
// both round each GEMV total to T, then compute `mlx_silu(gate) * up` with
// compiled silu (`x * sigmoid(x)` in T, sigmoid from MLX's `Sigmoid` functor).
// Default ON. `TRACK_SWIGLU_EPILOGUE=0` restores the two launches. Unexpected
// shapes also take the two-kernel path. This is not a §5.4 divergence.

import Foundation
import MLX

enum TrackSwiGLUEpilogue {
    /// Unset or any value other than `"0"` selects the fused GEMV epilogue.
    static let enabled =
        ProcessInfo.processInfo.environment["TRACK_SWIGLU_EPILOGUE"] != "0"

    struct Request: Equatable {
        var dtype: DType
        var tokens: Int
        var hidden: Int
        var intermediate: Int
        var bits: Int
        var groupSize: Int
        var sharedRows: Int
        var idxIsUInt32: Bool
        var xrowIsUInt32: Bool
    }

    static func canRunTwoKernel(_ r: Request) -> Bool {
        (r.dtype == .bfloat16 || r.dtype == .float16)
            && r.bits == 4
            && r.groupSize > 0
            && r.intermediate % 8 == 0
            && r.tokens >= 1 && r.tokens <= 8
            && r.sharedRows == 2 * r.intermediate
            && r.idxIsUInt32 && r.xrowIsUInt32
            && r.hidden > 0
            && r.hidden % r.groupSize == 0
    }

    /// The fused shared-expert walk uses `qmv_fast_reg` / `qmv_wide_reg_full`.
    static func canRunFused(_ r: Request) -> Bool {
        canRunTwoKernel(r) && TrackFastMoEKernels.isFast(k: r.hidden, n: r.intermediate)
    }

    static func shouldFuse(_ r: Request, enabled: Bool = TrackSwiGLUEpilogue.enabled) -> Bool {
        enabled && canRunFused(r)
    }

    /// Round float32 to bfloat16 (round to nearest, ties to even) and return
    /// the value as float32 with the low 16 bits cleared.
    static func roundToBFloat16(_ x: Float) -> Float {
        if x.isNaN {
            return Float(bitPattern: (x.bitPattern & 0xFFFF0000) | 0x007F0000)
        }
        if x.isInfinite { return x }
        let lsb = (x.bitPattern >> 16) & 1
        let (sum, overflow) = x.bitPattern.addingReportingOverflow(0x7FFF &+ lsb)
        if overflow { return Float.infinity }
        return Float(bitPattern: sum & 0xFFFF0000)
    }

    /// MLX `Sigmoid` functor in bfloat16: `y = 1/(1+exp(|x|))`, then `x<0 ? y : 1-y`.
    static func sigmoidBFloat16(_ x: Float) -> Float {
        let y = 1 / (1 + exp(abs(x)))
        return roundToBFloat16(x < 0 ? y : 1 - y)
    }

    /// Compiled silu in bfloat16: `x * sigmoid(x)`, each operation rounded to T.
    static func siluBFloat16(_ x: Float) -> Float {
        roundToBFloat16(x * sigmoidBFloat16(x))
    }

    /// Fused epilogue: `silu(gate) * up` after both halves are bfloat16.
    static func productBFloat16(_ gate: Float, _ up: Float) -> Float {
        let g = roundToBFloat16(gate)
        let u = roundToBFloat16(up)
        return roundToBFloat16(siluBFloat16(g) * u)
    }
}
