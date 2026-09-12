import MLX

/// Replays the existing mixer pair over the loaded immutable weight arrays.
final class TrackHCMixerReplay {
    private let standard: @Sendable ([MLXArray]) -> [MLXArray]
    private let withFloatInput: @Sendable ([MLXArray]) -> [MLXArray]

    init?(
        down: TrackProj, up: TrackProj, inject: TrackProj?, hcCount: Int, hidden: Int
    ) {
        guard case .quant(let dq) = down, case .quant(let uq) = up,
            dq.biases != nil, uq.biases != nil
        else { return nil }
        let iq: TrackQuantWeight?
        if let inject {
            guard case .quant(let q) = inject, q.biases != nil else { return nil }
            iq = q
        } else {
            iq = nil
        }
        standard = Self.make(
            down: dq, up: uq, inject: iq, hcCount: hcCount, hidden: hidden, emitF32: false)
        withFloatInput = Self.make(
            down: dq, up: uq, inject: iq, hcCount: hcCount, hidden: hidden, emitF32: true)
    }

    private static func make(
        down: TrackQuantWeight, up: TrackQuantWeight, inject: TrackQuantWeight?,
        hcCount: Int, hidden: Int, emitF32: Bool
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        compile(shapeless: false) { [down, up, inject] inputs in
            let normed = inputs[0]
            let S = normed.dim(1)
            let n2 = normed.reshaped(S, hcCount * hidden)
            let d = TrackFastMixerKernels.downInject(normed: n2, down: down, inject: inject)
            let u = TrackFastMixerKernels.upMix(
                act: d.act, normed: n2, up: up, inj: d.inj, hcCount: hcCount, hidden: hidden,
                hasInject: inject != nil, emitF32: emitF32)
            return [
                u.input.reshaped(1, S, hidden), u.inject.reshaped(1, S, hcCount), u.inputF32,
            ]
        }
    }

    /// The caller retains the original shape, stream, and debug guards.
    func call(_ normed: MLXArray, emitF32: Bool)
        -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray?)
    {
        let result = (emitF32 ? withFloatInput : standard)([normed])
        return (
            result[0], result[1],
            emitF32 ? result[2].reshaped(1, normed.dim(1), -1) : nil)
    }
}
