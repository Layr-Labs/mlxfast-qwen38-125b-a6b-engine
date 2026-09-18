// TrackPleGather.swift -- the PLE host row gather, overlapped with the GPU.
//
// WHAT THE SERIAL PATH PAID. The n-gram table is disk resident behind a
// bounded row cache, so `host.rows(globalIds:shape:)` is real work: a cold
// gather faults in sixteen small scattered rows per token through the
// MADV_RANDOM mapping, and even a warm gather is a locked walk of the LRU
// plus three buffer fills and the dequantize graph build. The serial path
// ran the whole chain -- token readback, context mirror, row-id hash, row
// gather -- inside `pleForward`, after the layer-0 block was already built.
// Every microsecond of it sat on the decode critical path with the GPU
// holding only what the first partial dispatch had already handed it.
//
// WHAT CHANGED. Everything the chain needs is known at the top of
// `fastStreams`: the fed tokens, the recurrent input state, the attention
// offset, and the context mirror. `TrackPleGather.start` runs the identical
// chain on a serial utility queue while the generation thread keeps
// building and dispatching the tower. `pleForward` joins on the box and
// receives exactly the arrays the inline code produced.
//
// ORDERING. The queue is serial, so gather blocks run in submission order
// and the context mirror -- a plain static -- is only ever mutated from
// this queue or from the generation thread while no gather is in flight.
// The table itself serializes gathers on its own lock, and the row source
// contract (`rows` returns the same values for the same ids) makes the
// result independent of when the call lands. The dequantize ops the gather
// builds bind to the same global GPU stream they bound to before; only the
// thread that enqueues the graph nodes changed.
//
// EXACTNESS. The chain below is the inline code moved, not rewritten: same
// dtype branch on the token readback, same mirror match/store/invalidate
// decisions, same `hostRowIds` hash, same `rows(globalIds:shape:)` call,
// same reshape and cast at the join. No value the layer consumes changes.

import Dispatch
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The result of one overlapped PLE host gather.
///
/// `@unchecked Sendable` because the box crosses from the generation thread
/// to the gather queue and back; the lock and the semaphore serialize every
/// access, and the payload is written once and read once.
final class TrackPleGatherBox: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var embedded: MLXArray?
    private var history: [Int64]?

    func deliver(embedded: MLXArray, history: [Int64]) {
        lock.lock()
        self.embedded = embedded
        self.history = history
        lock.unlock()
        semaphore.signal()
    }

    /// Blocks until the gather lands; the wait is the time the overlap did
    /// not hide, never longer than the serial gather it replaces.
    func join() -> (embedded: MLXArray, history: [Int64]) {
        semaphore.wait()
        lock.lock()
        defer { lock.unlock() }
        return (embedded!, history!)
    }
}

/// The immutable inputs one gather block reads on the queue.
///
/// `@unchecked Sendable` because `MLXArray` and the row-source existential
/// are not `Sendable`; every field is a `let` snapshot taken on the
/// generation thread and only ever read on the serial gather queue.
private final class TrackPleGatherInput: @unchecked Sendable {
    let embedding: Qwen4ExpNGramEmbedding
    let host: Qwen4ExpNGramHostRowSource
    let ids: MLXArray
    let ssm: MLXArray?
    let stateLayerIndex: Int
    let contextLength: Int
    let offset: Int
    let capture: Bool
    let eosTokenId: Int
    let nHeads: Int
    let box: TrackPleGatherBox

    init(
        embedding: Qwen4ExpNGramEmbedding, host: Qwen4ExpNGramHostRowSource,
        ids: MLXArray, ssm: MLXArray?, stateLayerIndex: Int, contextLength: Int,
        offset: Int, capture: Bool, eosTokenId: Int, nHeads: Int,
        box: TrackPleGatherBox
    ) {
        self.embedding = embedding
        self.host = host
        self.ids = ids
        self.ssm = ssm
        self.stateLayerIndex = stateLayerIndex
        self.contextLength = contextLength
        self.offset = offset
        self.capture = capture
        self.eosTokenId = eosTokenId
        self.nHeads = nHeads
        self.box = box
    }
}

enum TrackPleGather {

    /// One serial queue for every overlapped gather. Serial execution keeps
    /// the context-mirror mutations ordered exactly as the inline path had
    /// them, and the table's own gather lock serializes the row reads.
    private static let queue = DispatchQueue(label: "mlxfast.track.ple.gather")

    /// Starts the host gather for one PLE layer. Returns nil exactly when
    /// the inline path would take the device branch -- no host row source,
    /// or a window wider than the host path serves -- so `pleForward` keeps
    /// its fallback untouched.
    static func start(
        _ p: TrackPLE, ids: MLXArray,
        evaluation: CBv2RecurrentStateEvaluation,
        offset: Int, capture: Bool,
        eosTokenId: Int, nHeads: Int
    ) -> TrackPleGatherBox? {
        let S = ids.dim(1)
        guard let host = p.embedding.rowSourceHolder.source as? Qwen4ExpNGramHostRowSource,
            S <= 8
        else { return nil }

        let box = TrackPleGatherBox()
        let input = TrackPleGatherInput(
            embedding: p.embedding, host: host, ids: ids,
            ssm: evaluation.inputState(modelLayerIndex: p.stateLayerIndex)?.ssm,
            stateLayerIndex: p.stateLayerIndex,
            contextLength: max(1, p.dilation - 1),
            offset: offset, capture: capture,
            eosTokenId: eosTokenId, nHeads: nHeads, box: box)

        queue.async {
            let toks: [Int64] =
                input.ids.dtype == .int32
                ? input.ids.asArray(Int32.self).map(Int64.init)
                : input.ids.asType(.int64).asArray(Int64.self)
            let ctx: [Int64]
            if !input.capture,
                TrackPleContextMirror.matches(
                    offset: input.offset, layer: input.stateLayerIndex,
                    length: input.contextLength),
                let mirrored = TrackPleContextMirror.context
            {
                ctx = mirrored
            } else {
                ctx = input.ssm.map {
                    $0.dtype == .int32
                        ? $0.asArray(Int32.self).map(Int64.init)
                        : $0.asType(.int64).asArray(Int64.self)
                } ?? Array(repeating: Int64(input.eosTokenId), count: input.contextLength)
            }
            let history = ctx + toks
            if input.capture {
                TrackPleContextMirror.invalidate()
            } else {
                TrackPleContextMirror.store(
                    Array(history.suffix(input.contextLength)),
                    nextOffset: input.offset + S,
                    stateLayerIndex: input.stateLayerIndex,
                    contextLength: input.contextLength)
            }
            let gid = input.embedding.hostRowIds(history: [history], newCount: S)
            let rows = input.host.rows(globalIds: gid, shape: [1, S, input.nHeads])
            input.box.deliver(embedded: rows.reshaped(1, S, -1), history: history)
        }
        return box
    }
}
