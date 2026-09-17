// TrackPlePrefetch.swift -- window-entry prefetch of the PLE host block.
//
// The PLE layer's host path is the one host sync of a decode step: the fed
// ids and the staged context are read back, hashed on the host, and the
// n-gram rows are gathered out of the mmap'd table. All of it runs on the
// dispatch thread AFTER the previous window's GPU tail has drained, so the
// gather's host time is fully exposed.
//
// This prefetcher runs the same block on a serial helper queue, kicked at
// fastStreams entry before any of this window's graph is built. The ids
// readback then overlaps the previous window's GPU tail, and the hash +
// gather overlap this window's embed/layer-0 graph construction. The PLE
// layer awaits the finished job and consumes identical values: the same
// ids.asArray, the same mirror-or-ssm context, the same hostRowIds hash and
// the same host.rows gather, in the same order.
//
// Thread safety: the job reads only immutable inputs (the ids array, the
// bound recurrent input state, the context mirror, the row source). The
// mirror is written only by the dispatch thread after the job is awaited;
// the row gather serializes on the table's own gatherLock; MLXArray
// construction and lazy graph building are stream-safe.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

final class TrackPlePrefetch {
    /// The host block's outputs, identical to what `pleForward` computes
    /// inline when no prefetch is in flight.
    struct Result {
        /// `ctx + toks`: the staged context followed by the fed ids.
        let history: [Int64]
        /// `host.rows(globalIds:shape:)` output, before reshape/cast.
        let rows: MLXArray
    }

    private let queue = DispatchQueue(label: "track.ple.prefetch")
    private var pending: DispatchWorkItem?
    private var result: Result?

    /// Kick the host block for this window. Returns nil when the host row
    /// path does not apply (device row source or a window wider than the
    /// staged-context contract), in which case `pleForward` keeps its
    /// existing inline path.
    func start(
        ple: TrackPLE, ids: MLXArray, evaluation: CBv2RecurrentStateEvaluation,
        offset: Int, capture: Bool, eosTokenId: Int32, ngramHeads: Int
    ) -> Bool {
        let S = ids.dim(1)
        guard S <= 8,
            let host = ple.embedding.rowSourceHolder.source as? Qwen4ExpNGramHostRowSource
        else { return false }
        let contextLength = max(1, ple.dilation - 1)
        let stateLayerIndex = ple.stateLayerIndex
        let embedding = ple.embedding
        let work = DispatchWorkItem { [self] in
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
                let rawPrev = evaluation.inputState(modelLayerIndex: stateLayerIndex)?.ssm
                ctx =
                    rawPrev.map {
                        $0.dtype == .int32
                            ? $0.asArray(Int32.self).map(Int64.init)
                            : $0.asType(.int64).asArray(Int64.self)
                    } ?? Array(repeating: Int64(eosTokenId), count: contextLength)
            }
            let history = ctx + toks
            let gid = embedding.hostRowIds(history: [history], newCount: S)
            let rows = host.rows(globalIds: gid, shape: [1, S, ngramHeads])
            result = Result(history: history, rows: rows)
        }
        pending = work
        queue.async(execute: work)
        return true
    }

    /// Await the kicked job. Must be called at most once per `start`, from
    /// the dispatch thread, before the mirror is stored for this window.
    func awaitResult() -> Result? {
        guard let work = pending else { return nil }
        work.wait()
        pending = nil
        return result
    }
}
