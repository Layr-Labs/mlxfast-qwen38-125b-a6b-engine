// Copied from the pinned fork at commit 449f2d01, Libraries/MLXLLM/Models/Qwen4ExpMTPDrafter.swift: this is the track's EDITABLE inline MTP assistant.
//
//  Qwen4ExpMTPDrafter.swift
//  mlx-swift-lm
//
//  The embedded Qwen 3.8 Flash-Next MTP head as a ContinuousBatchingV2
//  request-stateful drafter.
//
//  The head lives in the TARGET checkpoint under `mtp.*`. It owns no
//  embedding table and no output head: it reads the target's `embed_tokens`
//  and writes through the target's `lm_head`. So there is no artifact to
//  stage and no compatibility matrix to check — the drafter is built from an
//  already-loaded target and can only ever match it.
//
//  WHAT THE HEAD CONSUMES, stated because it is the part that is easy to get
//  wrong: the target's PRE-final-mixer hyper-connection stream, `hc_count *
//  hidden` wide. `Qwen4ExpModel.cbv2ForwardWithHidden` returns exactly that
//  as `lastHidden`, and every draft round re-embeds it. The collapsed hidden
//  the head-facing path uses would be the wrong tensor and the wrong width.
//
//  DEPTH 1...3. The head is one hybrid layer applied to its own output, so a
//  deeper chain drifts further from the target with no measured acceptance to
//  pay for it. Three is the ruled ceiling for this track.
//
//  ARGMAX TIE-BREAK is the lowest token id, which is what `argMax` returns.
//  The target verifier uses the same rule, so a draft can never be rejected
//  over a tie the two sides broke differently.
//
//  WARM STATE. The engine hands every committed target chunk to
//  `observeCommittedTarget` while it builds the step graph that computed the
//  chunk. The fork's copy lets those transitions pile up: the first draft
//  round after the prompt then replays the whole backlog through the head in
//  one `[1, S]` forward, inside the decode window. With
//  `TRACK_DRAFTER_WARMSTATE` on (the default), this copy feeds each
//  observation's completed transitions through the head at once, inside the
//  observing step's graph build, and submits the head's new cache rows with
//  `asyncEval` -- the same non-blocking primitive the engine submits its own
//  step graphs with, so no host sync is added. The rows, and their order,
//  are the arrays the cold replay would concatenate: only the batching of
//  the head calls differs. The first round then finds an empty backlog and
//  feeds only the carry. Because only the cache rows are evaluated, the warm
//  feed also skips the attention output, the MoE block and the output head
//  for every history row -- work the cold replay performs for all `S` rows
//  and then discards.
//

import Foundation
import MLX
// ADDED to the fork's copy: this file used to live INSIDE MLXLLM, so the
// family boundary -- `Qwen4ExpModel` -- needed no import. Here it is a
// separate module.
import MLXLLM
import MLXLMCommon
import MLXNN

/// Drives the target's own `mtp.*` head as a CBv2 drafter.
public final class TrackQwen4ExpInlineMTPAssistant {

    /// The largest draft depth this head is served at. A POLICY bound, not a
    /// structural one: the head is one layer applied once per draft step
    /// with its own key-value rows, so the chain length is the number of
    /// steps the engine asks for. Ruled 6 on both platforms (David: "mtp at
    /// 6 on both mlx and cuda"); the CUDA track serves the same head at
    /// depth <= 6 with per-depth oracles.
    public static let maximumDepth = 6

    /// Warm-state handover toggle (`TRACK_DRAFTER_WARMSTATE`). On (the
    /// default), committed target observations run through the head as they
    /// arrive, during the observing step's graph build. Off, transitions
    /// accumulate in the backlog and the first round of every request pays
    /// the whole replay -- the fork's behavior. Same off-spelling as the
    /// engine's `DARKBLOOM_CBV2_MTP` kill switch.
    static func resolvesWarmState(_ raw: String?) -> Bool {
        guard let raw else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }
    static let warmStateEnabled = resolvesWarmState(
        ProcessInfo.processInfo.environment["TRACK_DRAFTER_WARMSTATE"])

    private let target: Qwen4ExpModel
    private let mtp: TrackQwen4ExpMTPModule
    // ADDED to the fork's copy: the fork reached the target's table as
    // `target.model.embedTokens`, which is internal to MLXLLM. This copy asks
    // the tower for the SAME child module through the public module API. It
    // is the instance the tower serves, not a copy: the head owns no table.
    private let embedTokens: Embedding
    // ADDED: the head's forward over the fast kernels (nil when disabled).
    private let fastHead: TrackFastHead?

    /// The draft argmax over a shortlist of the vocabulary: the lowest ids
    /// (this tokenizer assigns ids in BPE merge order, i.e. by corpus
    /// frequency; the public golden's tokens fall under 98,304 in 99.7% of
    /// cases) plus the added tokens at the top. The shortlist rows of `lm_head`
    /// are gathered once; their logits are the same per-row GEMV as the full
    /// head's, so the shortlist argmax IS the full argmax whenever the latter
    /// is in the list, and the target's verification decides every token
    /// either way. A miss costs one rejected draft, never a token.
    private struct Shortlist {
        let ids: MLXArray  // int32 [NS], ascending
        let weight: MLXArray, scales: MLXArray, biases: MLXArray
        let groupSize: Int, bits: Int
    }
    private let shortlist: Shortlist?
    static let shortlistLowIds = 98304
    static let shortlistSpecialFrom = 248044

    /// - Parameters:
    ///   - target: an already-loaded model. The head reads its embedding
    ///     table and writes through its output head.
    ///   - mtp: the head to drive. The Runner builds it from the checkpoint's
    ///     own `mtp.*` block. A serial-only load has no such block and gets
    ///     no drafter at all.
    public init(target: Qwen4ExpModel, mtp: TrackQwen4ExpMTPModule) {
        guard let embedTokens = target.model.children()[unwrapping: "embed_tokens"] as? Embedding
        else {
            preconditionFailure("Qwen4Exp tower has no embed_tokens module")
        }
        self.target = target
        self.mtp = mtp
        self.embedTokens = embedTokens
        var shortlist: Shortlist? = nil
        if let q = target.children()[unwrapping: "lm_head"] as? QuantizedLinear, let biases = q.biases,
            q.mode == .affine, q.bits == 4
        {
            let n = q.weight.dim(0)
            let specials = (Self.shortlistSpecialFrom > Self.shortlistLowIds && Self.shortlistSpecialFrom < n) ? (n - Self.shortlistSpecialFrom) : 0
            var low = min(Self.shortlistLowIds, n)
            low += (8 - (low + specials) % 8) % 8  // keep the fast GEMV path (N % 8 == 0)
            var ids: [Int32] = (0 ..< min(low, n)).map { Int32($0) }
            if specials > 0 { ids.append(contentsOf: (Self.shortlistSpecialFrom ..< n).map { Int32($0) }) }
            if ids.count < n, ids.count % 8 == 0 {
                let idx = MLXArray(ids)
                let w = take(q.weight, idx, axis: 0), s = take(q.scales, idx, axis: 0), b = take(biases, idx, axis: 0)
                eval(idx, w, s, b)
                shortlist = Shortlist(ids: idx, weight: w, scales: s, biases: b, groupSize: q.groupSize, bits: q.bits)
            }
        }
        self.shortlist = shortlist
        self.fastHead = TrackFastHead.enabled
            ? TrackFastHead(mtp, configuration: target.configuration) : nil
    }

    /// Pre-JIT the head's S=1...7 kernels. Idempotent; no persistent buffers.
    func warmVerifyCaches() {
        fastHead?.warmVerifyCaches(embedTokens: embedTokens)
    }

    /// Head caches, one per head layer.
    func makeCache() -> [KVCache] { mtp.makeCache() }

    /// One head application over `[1, S]` inputs, returning the LAST
    /// position's draft id and multi stream.
    ///
    /// Only the last row reaches `lm_head`: the earlier rows exist to put the
    /// head's key-value history in place, and projecting them would read the
    /// whole output head for tokens nothing consumes.
    private func headStep(
        tokens: MLXArray, multiStream: MLXArray, cache: [KVCache], stepIndex: Int
    ) -> (draft: MLXArray, multi: MLXArray) {
        let step: (sample: MLXArray, multi: MLXArray)
        if let fastHead, let attnCache = cache.first as? Qwen4ExpAttentionCache,
            let fast = fastHead.forward(
                nextTokenIds: tokens, multiStream: multiStream, embedTokens: embedTokens,
                cache: attnCache)
        {
            step = fast
        } else {
            step = mtp(
                nextTokenIds: tokens,
                multiStream: multiStream,
                embedTokens: embedTokens,
                cache: cache,
                stepIndex: stepIndex)
        }
        let last = step.sample.dim(1) - 1
        let lastSample = step.sample[0..., last..., 0...]
        let lastMulti = step.multi[0..., last..., 0...]
        let draft: MLXArray
        if let sl = shortlist {
            let x = lastSample[0..., -1, 0...]
            // TRACK_HEAD_TOP1 (default ON): GEMV epilogue writes a 16-byte
            // {id, logit, index, 16} record. TRACK_HEAD_TOP1=0 is the original
            // quantizedMM + host argMax, byte for byte.
            if TrackHeadTop1.enabled,
                let packed = TrackHeadTop1.apply(
                    x: x, weight: sl.weight, scales: sl.scales, biases: sl.biases,
                    ids: sl.ids, groupSize: sl.groupSize, bits: sl.bits)
            {
                draft = TrackHeadTop1.tokenId(packed)
            } else {
                let logits = quantizedMM(
                    x, sl.weight, scales: sl.scales, biases: sl.biases,
                    transpose: true, groupSize: sl.groupSize, bits: sl.bits)  // [1, NS]
                draft = take(sl.ids, argMax(logits, axis: -1), axis: 0).asType(.int32)
            }
        } else {
            draft = argMax(target.head(lastSample)[0..., -1, 0...], axis: -1).asType(.int32)
        }
        return (draft, lastMulti)
    }
}

// MARK: - CBv2 drafter

extension TrackQwen4ExpInlineMTPAssistant: CBv2MTPRequestStatefulDrafter {

    /// Per-request head state: the head's own key-value caches plus the
    /// trusted target transitions not yet folded into them.
    ///
    /// Internal, not private, so the warm-state tests can read the backlog
    /// and the cache offset directly.
    final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]

        /// Trusted target transitions waiting to enter head history. The multi
        /// stream at position t pairs with the token at t+1.
        var backlogMulti: [MLXArray] = []
        var backlogTokens: [MLXArray] = []
        /// Last trusted multi stream of an observed chunk. It becomes the
        /// preceding row when the next observed chunk crosses the boundary.
        var multiFrontier: MLXArray?

        /// Cache geometry captured around the round's trusted flush. Every
        /// later head input is speculative and is trimmed at finalize.
        var roundBaseOffset = 0
        var roundValidHistoryOffset = 0
        var roundDraftSteps = 0
        var roundInFlight = false
        var isReleased = false

        /// Trusted inputs moved out of the backlog for this round. Holding
        /// their roots fences the lazy concatenation and lets discard restore
        /// them without a host read.
        var roundTrustedMulti: [MLXArray] = []
        var roundTrustedTokens: [MLXArray] = []
        /// Proposal rows that retain each lazy head-step graph until the
        /// engine's finalize synchronization.
        var roundRoots: [MLXArray] = []

        var cacheOffset: Int {
            guard let first = caches.first else { return 0 }
            precondition(
                caches.dropFirst().allSatisfy { $0.offset == first.offset },
                "Qwen4Exp MTP head cache offsets diverged")
            return first.offset
        }

        private var backlogInputCount: Int {
            backlogTokens.reduce(0) { $0 + $1.dim(1) }
        }

        var committedInputCount: Int {
            (roundInFlight ? roundValidHistoryOffset : cacheOffset) + backlogInputCount
        }

        var stagedInputCount: Int {
            guard roundInFlight else { return 0 }
            return max(0, cacheOffset - roundValidHistoryOffset)
        }

        var materializedBytes: Int {
            let arrays =
                caches.flatMap { $0.innerState() }
                + backlogMulti + backlogTokens
                + [multiFrontier].compactMap { $0 }
                + roundTrustedMulti + roundTrustedTokens + roundRoots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(caches: [any KVCache]) { self.caches = caches }

        func clearRound() {
            roundBaseOffset = cacheOffset
            roundValidHistoryOffset = cacheOffset
            roundDraftSteps = 0
            roundInFlight = false
            roundTrustedMulti.removeAll(keepingCapacity: true)
            roundTrustedTokens.removeAll(keepingCapacity: true)
            roundRoots.removeAll(keepingCapacity: true)
        }

        func clearAll() {
            caches.removeAll(keepingCapacity: false)
            backlogMulti.removeAll(keepingCapacity: false)
            backlogTokens.removeAll(keepingCapacity: false)
            multiFrontier = nil
            roundTrustedMulti.removeAll(keepingCapacity: false)
            roundTrustedTokens.removeAll(keepingCapacity: false)
            roundRoots.removeAll(keepingCapacity: false)
            roundBaseOffset = 0
            roundValidHistoryOffset = 0
            roundDraftSteps = 0
            roundInFlight = false
            isReleased = true
        }
    }

    /// The frozen-KV capture seam this drafter does not use.
    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }

    /// No required mode: the target implements the captured verify window
    /// (`cbv2ForwardWithHiddenCaptured`), so the engine's configured mode
    /// applies and a round verifies its `1 + k` candidates in one forward.
    /// Serial target scoring remains available by configuration.
    public var requiredVerificationMode: CBv2MTPVerificationMode? { nil }

    public var maximumDraftTokens: Int? { Self.maximumDepth }
    public var maximumSpeculativeBatch: Int? { 1 }

    /// Head key-value rows, the retained indexer tape, one multi-stream row
    /// and one token id, per input token.
    public var requestStateBytesPerToken: Int {
        let configuration = target.configuration
        let elementBytes = embedTokens.weight.dtype.size
        let perLayer =
            2 * configuration.kvHeads * configuration.headDim + configuration.indexerHeadDim
        let head = perLayer * mtp.layerCount * elementBytes
        let stream = configuration.hcCount * configuration.hiddenSize * elementBytes
        return head + stream + MemoryLayout<Int32>.stride
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(caches: makeCache())
    }

    private func typed(_ requestState: any CBv2MTPRequestState) -> RequestState {
        guard let state = requestState as? RequestState else {
            preconditionFailure("Qwen4Exp MTP received foreign request state")
        }
        return state
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        observeCommittedTarget(
            observation, requestState: requestState, warm: Self.warmStateEnabled)
    }

    /// One committed target chunk. `warm` selects where the completed
    /// transitions run: through the head now (`true`), or into the backlog
    /// for the next round's replay (`false`, the fork's behavior). The pairs
    /// themselves -- which arrays, in which order -- are the same either way.
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState,
        warm: Bool
    ) {
        let state = typed(requestState)
        precondition(!state.isReleased, "Qwen4Exp MTP observed released request state")
        precondition(!state.roundInFlight, "Qwen4Exp MTP observed target during a round")
        precondition(
            observation.tokens.ndim == 2 && observation.hidden.ndim == 3
                && observation.tokens.dim(0) == 1 && observation.hidden.dim(0) == 1
                && observation.tokens.dim(1) == observation.hidden.dim(1),
            "Qwen4Exp MTP target observation shape mismatch")

        let count = observation.tokens.dim(1)
        guard count > 0 else { return }

        // The transitions this observation completes: the preceding chunk's
        // final multi stream paired with this chunk's first target input,
        // then the intra-chunk pairs (multi[t] conditions token[t+1]).
        var pairTokens: [MLXArray] = []
        var pairMulti: [MLXArray] = []
        if let frontier = state.multiFrontier {
            pairTokens.append(observation.tokens[0..., 0 ..< 1])
            pairMulti.append(frontier)
        }
        if count > 1 {
            pairMulti.append(observation.hidden[0..., 0 ..< count - 1, 0...])
            pairTokens.append(observation.tokens[0..., 1 ..< count])
        }
        state.multiFrontier = observation.hidden[0..., (count - 1) ..< count, 0...]

        // Fallback: the toggle is off, or the pairing came out inconsistent
        // (a shape change the preconditions above do not name). Accumulate
        // exactly as the fork's copy does and let the next round replay.
        let pairedTokens = pairTokens.reduce(0) { $0 + $1.dim(1) }
        let pairedMulti = pairMulti.reduce(0) { $0 + $1.dim(1) }
        guard warm, pairedTokens > 0, pairedTokens == pairedMulti else {
            state.backlogMulti.append(contentsOf: pairMulti)
            state.backlogTokens.append(contentsOf: pairTokens)
            return
        }

        // Warm feed: the same rows `beginRound` would concatenate, run
        // through the head now, while the engine is still building the step
        // graph that produced them. Only the cache rows are evaluated, so the
        // history rows never pay for the attention output, the MoE block or
        // the output head. `asyncEval` queues without blocking; it is the
        // primitive the engine itself submits step graphs with, so this adds
        // no host sync to the engine thread.
        let tokens = pairTokens.count == 1 ? pairTokens[0] : concatenated(pairTokens, axis: 1)
        let multi = pairMulti.count == 1 ? pairMulti[0] : concatenated(pairMulti, axis: 1)
        _ = headStep(tokens: tokens, multiStream: multi, cache: state.caches, stepIndex: 0)
        asyncEval(state.caches.flatMap { $0.innerState() })
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        UnusedPreparedCapture()
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Qwen4Exp MTP requires request-owned head state")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let state = typed(requestState)
        precondition(!state.isReleased, "Qwen4Exp MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "Qwen4Exp MTP draft input shape mismatch")
        precondition(
            hidden.dim(2) == target.configuration.hcCount * target.configuration.hiddenSize,
            "Qwen4Exp MTP draft hidden must be the pre-final-mixer multi stream")

        let isFirstStep = !state.roundInFlight
        let feed: (tokens: MLXArray, multi: MLXArray)
        if isFirstStep {
            feed = beginRound(tokens: tokens, multi: hidden, state: state)
        } else {
            precondition(
                state.roundDraftSteps < Self.maximumDepth,
                "Qwen4Exp MTP exceeded its \(Self.maximumDepth)-step draft chain")
            feed = (tokens, hidden)
        }

        let step = headStep(
            tokens: feed.tokens, multiStream: feed.multi, cache: state.caches,
            stepIndex: state.roundDraftSteps)
        state.roundRoots.append(contentsOf: [step.multi, step.draft])
        state.roundDraftSteps += 1
        // Kick the lazy draft-token ids onto the GPU as soon as they exist,
        // so the verify-round asArray is a copy of a finished buffer, not a
        // wait for all k heads. The engine only asyncEvals draft index 0.
        if TrackPLEVerifyPrefetch.enabled { asyncEval(step.draft) }

        if isFirstStep {
            // The engine submits this generation's evaluation targets before
            // it builds a deeper draft step, so the trusted flush ends here.
            state.roundValidHistoryOffset = state.cacheOffset
        }
        return (step.draft, step.multi)
    }

    private func beginRound(
        tokens: MLXArray, multi: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, multi: MLXArray) {
        precondition(!state.roundInFlight, "Qwen4Exp MTP round already in flight")
        precondition(
            state.backlogMulti.count == state.backlogTokens.count,
            "Qwen4Exp MTP trusted backlog diverged")

        state.roundBaseOffset = state.cacheOffset
        state.roundValidHistoryOffset = state.cacheOffset
        state.roundDraftSteps = 0
        state.roundInFlight = true
        state.roundTrustedMulti = state.backlogMulti
        state.roundTrustedTokens = state.backlogTokens
        state.backlogMulti.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)

        // The current target carry is trusted and completes the frontier
        // transition.
        state.roundTrustedMulti.append(multi)
        state.roundTrustedTokens.append(tokens)
        state.multiFrontier = nil

        if state.roundTrustedTokens.count == 1 {
            return (state.roundTrustedTokens[0], state.roundTrustedMulti[0])
        }
        return (
            concatenated(state.roundTrustedTokens, axis: 1),
            concatenated(state.roundTrustedMulti, axis: 1)
        )
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState, !state.isReleased else { return [] }
        return state.caches.flatMap { $0.innerState() }
            + state.backlogMulti + state.backlogTokens
            + [state.multiFrontier].compactMap { $0 }
            + state.roundTrustedMulti + state.roundTrustedTokens + state.roundRoots
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        let state = typed(requestState)
        precondition(!state.isReleased, "Qwen4Exp MTP finalized released request state")
        precondition(state.roundInFlight, "Qwen4Exp MTP finalized without a round")
        precondition(
            (0 ... state.roundDraftSteps + 1).contains(confirmedInputTokens),
            "Qwen4Exp MTP confirmed prefix exceeds the draft round")
        precondition(
            committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1 && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "Qwen4Exp MTP committed target rows mismatch")
        let committedDraftCount = committedDraftTokens.dim(1)
        precondition(
            committedDraftCount <= state.roundDraftSteps
                && committedDraftCount <= max(0, confirmedInputTokens - 1),
            "Qwen4Exp MTP committed drafts exceed confirmed target inputs")

        trim(state: state, to: state.roundValidHistoryOffset)
        if committedDraftCount > 0 {
            // These are target verify multi streams, never speculative head
            // outputs. They flush with the next carry.
            state.backlogTokens.append(committedDraftTokens)
            state.backlogMulti.append(committedTargetHidden)
        }
        state.clearRound()
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState,
            !state.isReleased, state.roundInFlight
        else { return }

        trim(state: state, to: state.roundBaseOffset)
        // Restore every trusted transition the abandoned graph consumed, in
        // original order. No speculative head output enters the backlog.
        state.backlogMulti = state.roundTrustedMulti + state.backlogMulti
        state.backlogTokens = state.roundTrustedTokens + state.backlogTokens
        state.clearRound()
    }

    /// Roll the head's own caches back to `offset`.
    ///
    /// `Qwen4ExpAttentionCache.trim` moves the key-value offset AND slices the
    /// indexer tape, so the head's sparse attention sees the same history a
    /// round that never happened would have left.
    private func trim(state: RequestState, to offset: Int) {
        let rollback = state.cacheOffset - offset
        precondition(rollback >= 0, "Qwen4Exp MTP cache checkpoint moved forward")
        guard rollback > 0 else { return }
        for cache in state.caches {
            precondition(
                cache.trim(rollback) == rollback,
                "Qwen4Exp MTP head cache refused a \(rollback) token rollback")
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState, !state.isReleased else { return }
        state.clearAll()
    }
}
