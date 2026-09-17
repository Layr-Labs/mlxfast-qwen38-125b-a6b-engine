// TrackPLEWindowPump.swift -- stage the PLE window's host inputs off the
// generation thread.
//
// WHAT THE INLINE PATH DOES, per small window, at the moment the layer loop
// reaches the PLE layer (TrackFastModel.pleForward):
//
//   ids.asArray            one device readback of the fed token ids
//   state.ssm.asArray      a second readback whenever the context mirror
//                          cannot serve the rolling n-gram context
//   hostRowIds             host-side hash of the context ++ tokens history
//   host.rows              the n-gram table gather: LRU hits are memcpy,
//                          misses are serial MADV_RANDOM mmap page-ins
//
// All of it is host work that depends only on the window's ids and the
// committed context -- both known the instant the window opens, before the
// layer loop dispatches its first kernel. Running it inline parks the
// generation thread inside the layer loop: the id readback waits on the
// draft chain's evaluation, and every page-in is serial time no GPU work
// covers.
//
// THE PUMP. `stage` is called at window open, before the layer loop. A
// serial queue performs the readbacks, the hash and the gather while the
// generation thread dispatches layer 0 and beyond; `join` at the PLE layer
// hands over the finished `embedded` and `history`. The gather goes through
// the same public `rows(globalIds:shape:)` the inline path calls, so the
// served bytes are identical -- only the thread and the timing of the
// page-ins change. The mirror store stays on the generation thread inside
// `pleForward`, so every mirror mutation keeps its existing serialization.
//
// `join` consumes the staged value: a window that never reaches the PLE
// layer leaves nothing behind, and a stale stage can never be served twice.

import Foundation
import MLX
import MLXLLM

enum TrackPLEWindowPump {
    struct Staged {
        let embedded: MLXArray
        let history: [Int64]
    }

    nonisolated(unsafe) private static var staged: Staged? = nil
    private static let queue = DispatchQueue(
        label: "TrackPLEWindowPump", qos: .userInteractive)

    /// Begin the window's host staging. `ctx` is the rolling n-gram context
    /// resolved on the generation thread (mirror hit, or the one state read
    /// a miss still costs); everything after it moves to the pump queue.
    static func stage(
        ids: MLXArray, ple: TrackPLE, host: any Qwen4ExpNGramHostRowSource,
        ctx: [Int64], S: Int, B: Int, rowCount: Int, dtype: DType
    ) {
        queue.async {
            let toks: [Int64] =
                ids.dtype == .int32
                ? ids.asArray(Int32.self).map(Int64.init)
                : ids.asType(.int64).asArray(Int64.self)
            let history = ctx + toks
            let gid = ple.embedding.hostRowIds(history: [history], newCount: S)
            let rows = host.rows(globalIds: gid, shape: [B, S, rowCount])
            staged = Staged(
                embedded: rows.reshaped(B, S, -1).asType(dtype), history: history)
        }
    }

    /// Hand the staged window to the PLE layer. Consumes the value either
    /// way, so a window that skipped staging or never reached the layer
    /// cannot leak a stale result into a later window.
    static func join() -> Staged? {
        queue.sync { }
        let value = staged
        staged = nil
        return value
    }
}
