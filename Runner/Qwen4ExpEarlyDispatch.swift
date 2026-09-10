// Qwen4ExpEarlyDispatch.swift
//
// A bounded host-side scheduling experiment for the Qwen 3.8 target.
//
// The pinned model keeps its CBv2 decoder loop inside MLXLLM. That loop is not
// visible to this editable Runner, so this file uses the smallest public seam:
// replace the target decoder's late attention projections with subclasses that
// run the original quantized operation, then submit its result to MLX while the
// host builds the rest of the layer. The tensor tree and the math stay the
// same. The dispatch is used only for decode and verify sized inputs.

import Foundation
import MLX
import MLXLLM
import MLXNN

/// Bounded eager dispatch for target decoder attention outputs.
public enum TrackQwen4ExpEarlyDispatch {

    /// The row bound used by the oMLX scheduling experiment.
    public static let maxRows = 64

    /// The process-level kill switch. It is read once when projections are
    /// installed, then captured by each wrapper.
    public static let environmentVariable = "MLXFAST_QWEN4_EARLY_DISPATCH"

    /// Returns false for prefill-sized inputs and for the explicit kill switch.
    public static func shouldDispatch(rowCount: Int, enabled: Bool) -> Bool {
        enabled && rowCount > 0 && rowCount <= maxRows
    }

    /// Read the kill switch. The default is enabled for the experiment.
    public static func environmentEnabled() -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[environmentVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        else { return true }
        return !["0", "false", "no", "off"].contains(raw)
    }

    /// Projection paths that end the attention branch of one target decoder
    /// layer. MTP-head projections are deliberately outside this prefix.
    public static func isTargetAttentionProjection(path: String) -> Bool {
        guard path.hasPrefix("model.layers.") else { return false }
        return path.hasSuffix(".self_attn.o_proj")
            || path.hasSuffix(".linear_attn.out_proj")
    }

    /// Replace the 48 target attention output projections with exact
    /// quantized subclasses. The returned count is diagnostic only.
    @discardableResult
    public static func install(on model: Qwen4ExpModel, enabled: Bool? = nil) -> Int {
        let dispatchEnabled = enabled ?? environmentEnabled()
        guard dispatchEnabled else { return 0 }

        var replacements: [(String, Module)] = []
        for (path, module) in model.leafModules().flattened()
        where isTargetAttentionProjection(path: path)
        {
            guard let quantized = module as? QuantizedLinear,
                !(module is TrackQwen4ExpEarlyDispatchQuantizedLinear)
            else { continue }
            replacements.append(
                (
                    path,
                    TrackQwen4ExpEarlyDispatchQuantizedLinear(
                        quantized, enabled: dispatchEnabled)))
        }

        guard !replacements.isEmpty else { return 0 }
        model.update(modules: ModuleChildren.unflattened(replacements))
        return replacements.count
    }

    /// Submit a projection result without changing its value or dtype.
    @inline(__always)
    static func submit(_ output: MLXArray, input: MLXArray, enabled: Bool) {
        guard input.ndim >= 2 else { return }
        // Decoder projections receive [B, S, K]. Keep the 2-D form useful for
        // small module tests and generic callers: its first axis is rows.
        let rows = input.ndim >= 3 ? input.dim(0) * input.dim(1) : input.dim(0)
        guard shouldDispatch(rowCount: rows, enabled: enabled) else { return }
        asyncEval(output)
    }
}

/// QuantizedLinear subclass used only at the target's attention output seam.
/// All parameter arrays and quantization metadata are the original objects.
public final class TrackQwen4ExpEarlyDispatchQuantizedLinear: QuantizedLinear {
    private let dispatchEnabled: Bool

    public init(_ original: QuantizedLinear, enabled: Bool = true) {
        self.dispatchEnabled = enabled
        super.init(
            weight: original.weight,
            bias: original.bias,
            scales: original.scales,
            biases: original.biases,
            groupSize: original.groupSize,
            bits: original.bits,
            mode: original.mode)
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = super.callAsFunction(x)
        TrackQwen4ExpEarlyDispatch.submit(
            output, input: x, enabled: dispatchEnabled)
        return output
    }
}
