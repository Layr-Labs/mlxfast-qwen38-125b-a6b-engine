// TrackPlePrefetch.swift -- off-thread host row gather for the PLE layer.
//
// WHAT THIS IS. The PLE layer's host path resolves three serial host steps on
// the decode critical path: read the fed tokens back to the host, hash them
// into n-gram row ids, and gather those rows from the SSD-backed table (LRU
// memcpy or mmap page faults). All of it depends only on `ids` and the staged
// context -- both known the moment `fastStreams` is entered, one layer before
// the PLE block runs. `TrackPleGather` runs that chain on a side thread so it
// overlaps the graph construction of the layers ahead of PLE (and any GPU
// tail the token readback waits on) instead of idling the decode loop inside
// `pleForward`.
//
// Thread-safety: the side thread touches only the token array (an input,
// already produced), the staged `state.ssm` (produced by the previous step),
// the context mirror (no other accessor runs concurrently -- the main thread
// is between `fastStreams` entry and the `pleForward` join), and the row
// source (internally serialized by its own gather lock). `evaluation` is
// never touched off-thread: `inputState` is resolved on the caller's thread
// before the spawn. The produced `rows` is a lazy MLXArray joined back on the
// caller's thread; nothing evaluates on the side thread except the readbacks
// the inline path would have performed there anyway.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// One in-flight host gather for a single `fastStreams` call.
final class TrackPleGather {
    private var result: (embedded: MLXArray, history: [Int64])?
    private let done = DispatchSemaphore(value: 0)

    /// Starts the host chain on a dedicated serial queue. `state` must be
    /// resolved by the caller on its own thread (`evaluation` is not
    /// thread-safe); everything else is read-only or internally locked.
    static func start(
        embedding: Qwen4ExpNGramEmbedding,
        host: Qwen4ExpNGramHostRowSource,
        ids: MLXArray,
        state: CBv2RecurrentLayerState?,
        offset: Int,
        S: Int,
        capture: Bool,
        contextLength: Int,
        stateLayerIndex: Int,
        eosTokenId: Int,
        ngramHeads: Int
    ) -> TrackPleGather {
        let gather = TrackPleGather()
        TrackPleGather.queue.async {
            gather.run(
                embedding: embedding, host: host, ids: ids, state: state,
                offset: offset, S: S, capture: capture,
                contextLength: contextLength, stateLayerIndex: stateLayerIndex,
                eosTokenId: eosTokenId, ngramHeads: ngramHeads)
        }
        return gather
    }


    private static let queue = DispatchQueue(label: "track.ple.gather")

    /// The same host chain `pleForward` runs inline when no gather is in
    /// flight: token readback, context resolve (mirror or staged ssm), mirror
    /// bookkeeping, row-id hash, row gather.
    private func run(
        embedding: Qwen4ExpNGramEmbedding,
        host: Qwen4ExpNGramHostRowSource,
        ids: MLXArray,
        state: CBv2RecurrentLayerState?,
        offset: Int,
        S: Int,
        capture: Bool,
        contextLength: Int,
        stateLayerIndex: Int,
        eosTokenId: Int,
        ngramHeads: Int
    ) {
        defer { done.signal() }
        let toks: [Int64] =
            ids.dtype == .int32
            ? ids.asArray(Int32.self).map(Int64.init)
            : ids.asType(.int64).asArray(Int64.self)
        let ctx: [Int64]
        if !capture,
            TrackPleContextMirror.matches(
                offset: offset, layer: stateLayerIndex, length: contextLength),
            let mirrored = TrackPleContextMirror.context
        {
            ctx = mirrored
        } else {
            ctx =
                state?.ssm.map {
                    $0.dtype == .int32
                        ? $0.asArray(Int32.self).map(Int64.init)
                        : $0.asType(.int64).asArray(Int64.self)
                } ?? Array(repeating: Int64(eosTokenId), count: contextLength)
        }
        let history = ctx + toks
        if capture {
            TrackPleContextMirror.invalidate()
        } else {
            TrackPleContextMirror.store(
                Array(history.suffix(contextLength)), nextOffset: offset + S,
                stateLayerIndex: stateLayerIndex, contextLength: contextLength)
        }
        let gid = embedding.hostRowIds(history: [history], newCount: S)
        let rows = host.rows(globalIds: gid, shape: [1, S, ngramHeads])
        result = (rows.reshaped(1, S, -1), history)
    }

    /// Blocks until the side thread finishes. Returns the gathered rows
    /// (pre-dtype-cast) and the resolved host history, exactly what the inline
    /// path produces.
    func join() -> (embedded: MLXArray, history: [Int64]) {
        done.wait()
        if let result { return result }
        preconditionFailure("TrackPleGather: side gather produced no result")
    }
}
