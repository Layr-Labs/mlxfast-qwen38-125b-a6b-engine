import MLX
import MLXRunners

/// Keep session state alive until its outstanding GPU work has completed.
/// The single-row stepper asynchronously evaluates recurrent and cache roots
/// alongside logits; reading the logits alone need not join every root.
final class TrackSynchronizedStepper: TeacherForcedStepper {
    private let inner: any TeacherForcedStepper
    private let stream: MLX.Stream

    init(_ inner: any TeacherForcedStepper, stream: MLX.Stream) {
        self.inner = inner
        self.stream = stream
    }

    deinit {
        stream.synchronize()
    }

    func begin() throws {
        if inner.forwards > 0 { stream.synchronize() }
        try inner.begin()
    }

    func forward(_ tokens: [Int]) throws -> StepOutput {
        try inner.forward(tokens)
    }

    var forwards: Int { inner.forwards }
}
