// TrackPleHistoryCache.swift -- chained host context for the PLE n-gram path.
//
// The context mirror serves the next step only when nothing speculative ran:
// every capture-verify window invalidates it, so the following window falls
// back to reading the staged ssm state back to the host -- a device sync in
// the middle of the host gather path. The context a capture step needs is
// already on the host: the previous window's `ctx + toks` history covers
// every offset the commit can produce, because acceptance advances the
// offset by at most the window width. This cache keeps that one history and
// slices the context for any offset inside it, so chained verify rounds and
// the first window after a decode step never touch the device for context.
//
// Single slot, same single-stream assumption as TrackPleContextMirror: a
// request whose offset does not continue the stored history misses and takes
// the existing fallback. Offsets are monotonic within a stream, so a stale
// entry can only miss, never serve wrong tokens.

enum TrackPleHistoryCache {
    nonisolated(unsafe) private static var history: [Int64] = []
    nonisolated(unsafe) private static var baseOffset: Int = 0  // token position of history[0]
    nonisolated(unsafe) private static var layer: Int = -1

    /// The `length` tokens ending at `offset`, when the stored history covers
    /// them. Returns nil on any gap, layer mismatch, or empty store.
    static func context(offset: Int, layer: Int, length: Int) -> [Int64]? {
        guard layer == self.layer, !history.isEmpty else { return nil }
        let k = offset - length - baseOffset
        guard k >= 0, k + length <= history.count else { return nil }
        return Array(history[k ..< k + length])
    }

    /// Record one window's `ctx + toks`; `endOffset` is the position one past
    /// the last token (the offset the next step uses if all of them commit).
    static func store(_ newHistory: [Int64], endOffset: Int, layer: Int) {
        history = newHistory
        baseOffset = endOffset - newHistory.count
        self.layer = layer
    }

    static func invalidate() {
        history = []
        layer = -1
    }
}
