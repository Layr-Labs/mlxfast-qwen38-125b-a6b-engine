import Foundation
import MLX

/// Serial side queue that warms the PLE n-gram row cache while the engine is
/// still in the MTP draft phase of a round.
///
/// The verify forward pays serial CPU time inside its graph build for the
/// n-gram row fetch: a hash of the committed history, then a bounded-LRU
/// lookup that falls through to a mapped-file page fault on a miss. The draft
/// loop produces the very tokens the verify window will hash — every drafted
/// token occupies a position in that window whether or not it is accepted —
/// so the row read for each position can be queued as the draft step that
/// produces it returns, and overlaps the remaining draft steps and the verify
/// graph build.
///
/// All cache mutation from the warm path is funneled through one serial
/// queue, which also carries the rolling token history: each queued block
/// reads its draft token back, appends it, and warms the rows of the position
/// it closes. The forward calls `drain()` immediately before its own row
/// lookup, which is a no-op when the queued warms already landed and
/// otherwise waits for the in-flight read — never longer than the
/// synchronous fetch it replaces.
final class TrackPleWarmer: @unchecked Sendable {
    static let shared = TrackPleWarmer()

    private let queue = DispatchQueue(label: "track.ple.warmer", qos: .userInitiated)

    /// Rolling token history of the in-flight round, owned by the queue.
    /// Seeded by `startRound` and extended one drafted token per
    /// `appendDraft`.
    private var history: [Int64] = []

    /// Round start. `seedTokens` is the round's trusted token carry — the
    /// committed suffix plus the carry token. The verify window re-feeds the
    /// carry, so the hash history of its first position is the last two
    /// committed tokens plus the carry again: `suffix(2) ++ [last]`. The
    /// readback runs on the warm queue, off the building thread. `warm`
    /// receives the seeded history and fetches its rows.
    func startRound(seedTokens: MLXArray, warm: @escaping ([Int64]) -> Void) {
        queue.async {
            let ids = seedTokens.asType(.int32).asArray(Int32.self).map(Int64.init)
            guard let last = ids.last else { return }
            self.history = Array(ids.suffix(2)) + [last]
            warm(self.history)
        }
    }

    /// One drafted token. The readback runs on the warm queue, so the draft
    /// step's graph is evaluated while the calling thread keeps building —
    /// the same overlap the row fetch gets. `warm` receives the history
    /// ending at this token and fetches the rows of the position it closes.
    func appendDraft(_ draft: MLXArray, warm: @escaping ([Int64]) -> Void) {
        queue.async {
            self.history.append(Int64(draft.asType(.int32).asArray(Int32.self)[0]))
            warm(self.history)
        }
    }

    /// Wait until every block queued so far has finished. Called on the
    /// forward thread just before the row lookup so the lookup observes all
    /// warmed rows and never races an in-flight warm.
    func drain() {
        queue.sync {}
    }
}
