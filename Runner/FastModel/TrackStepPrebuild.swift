// TrackStepPrebuild.swift -- speculative next-step host graph during MTP verify.
//
// SEAM. The draft/verify/accept loop lives in vendor EngineLoopV2. After it
// submits step N's captured verify, the host is idle until the acceptance
// packet is read at finalize. Runner cannot intercept finalize. The hook we
// own is the captured verify forward: once that graph is built (asyncChunk
// has already submitted earlier layers), construct step N+1 assuming
// keepPositions = window width — the common accept-all case. Full accept
// submits that graph; partial/reject drops it and restores the accepted
// boundary from the w26 snapshot bank (TrackGDNSnapshotBank /
// TrackGDNStateStore). Building is host-side. Nothing is enqueued that
// depends on unverified tokens. A wait-on-acceptance Metal/MLX fence could
// enqueue the pre-built graph early; this branch does not.
//
// TRACK_STEP_PREBUILD=0, width-1, or TRACK_GDN_STATE_RESTORE=0 keep today's
// build-on-accept path. No second snapshot bank.

import Foundation

public enum TrackStepPrebuild {
    /// `TRACK_STEP_PREBUILD=0` falls back to build-on-accept.
    public static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_STEP_PREBUILD"] ?? "1") != "0"
    }()

    /// Needs the w26 prefix-replay bank (width 2+ and restore on).
    public static func shouldPrebuild(
        positions: Int,
        prebuildEnabled: Bool = TrackStepPrebuild.enabled,
        restoreEnabled: Bool = TrackGDNStateRestore.enabled
    ) -> Bool {
        prebuildEnabled && restoreEnabled && positions >= 2
    }
}

public enum TrackStepPrebuildOutcome: Equatable, Sendable {
    /// Accept-all: the graph built during verify is submitted now.
    case submittedPrebuilt
    /// Partial/reject: pre-build dropped (never enqueued), bank restored,
    /// corrected step built after acceptance.
    case discardedAndRebuilt
    /// Width-1, toggle off, or restore off: today's build-on-accept path.
    case fallbackBuildAfterAccept
}

/// Host-side decode-step graph for the assumed accept-all next step.
/// `startState` is a copy of the bank's last slot, so a later snapshot into
/// that slot cannot alias this graph. `enqueued` stays false until acceptance.
public struct TrackPrebuiltDecodeGraph: Equatable, Sendable {
    public let identity: Int
    public let assumedKeep: Int
    public let startSlot: Int
    public let startState: [TrackGDNMockState]
    public var enqueued: Bool

    public init(
        identity: Int, assumedKeep: Int, startSlot: Int,
        startState: [TrackGDNMockState], enqueued: Bool
    ) {
        self.identity = identity
        self.assumedKeep = assumedKeep
        self.startSlot = startSlot
        self.startState = startState
        self.enqueued = enqueued
    }
}

/// CPU-testable schedule: verify submit → pre-build N+1 → acceptance →
/// submit or discard+restore+rebuild. Reuses `TrackGDNSnapshotBank`.
public final class TrackStepPrebuildScheduler: @unchecked Sendable {
    public let bank: TrackGDNSnapshotBank
    public private(set) var events: [String] = []
    public private(set) var prebuilt: TrackPrebuiltDecodeGraph?
    public private(set) var hostBuildCount = 0
    public private(set) var lastOutcome: TrackStepPrebuildOutcome?

    private let prebuildEnabled: Bool
    private let restoreEnabled: Bool
    private var nextIdentity = 1
    private var verifyWidth = 0

    public init(
        bank: TrackGDNSnapshotBank,
        prebuildEnabled: Bool = TrackStepPrebuild.enabled,
        restoreEnabled: Bool = TrackGDNStateRestore.enabled
    ) {
        self.bank = bank
        self.prebuildEnabled = prebuildEnabled
        self.restoreEnabled = restoreEnabled
    }

    /// Call after step N's verify graph has been submitted (GPU in flight).
    /// Snapshots for the window must already sit in `bank`.
    public func afterVerifySubmitted(width: Int) {
        events.append("verifySubmit")
        verifyWidth = width
        prebuilt = nil
        lastOutcome = nil
        guard TrackStepPrebuild.shouldPrebuild(
            positions: width, prebuildEnabled: prebuildEnabled,
            restoreEnabled: restoreEnabled)
        else { return }
        buildPrebuilt(assumedKeep: width)
    }

    /// Acceptance packet arrived. Full accept submits the pre-build (its
    /// start-state copy, not a re-read of the bank). Partial drops the
    /// pre-build without enqueue, restores the accepted boundary from the
    /// bank, and builds the corrected step.
    @discardableResult
    public func onAcceptance(
        confirmed: Int, live: inout [TrackGDNMockState]
    ) -> TrackStepPrebuildOutcome {
        events.append("acceptance")
        if let pre = prebuilt {
            precondition(
                !pre.enqueued,
                "TrackStepPrebuild: pre-built graph must not enqueue before acceptance")
            if confirmed == pre.assumedKeep, confirmed == verifyWidth {
                events.append("submitPrebuilt")
                for layer in live.indices {
                    live[layer] = pre.startState[layer]
                }
                prebuilt = TrackPrebuiltDecodeGraph(
                    identity: pre.identity, assumedKeep: pre.assumedKeep,
                    startSlot: pre.startSlot, startState: pre.startState, enqueued: true)
                lastOutcome = .submittedPrebuilt
                return .submittedPrebuilt
            }
            events.append("discard")
            prebuilt = nil
            restoreLive(&live, confirmed: confirmed)
            events.append("rebuild")
            hostBuildCount += 1
            lastOutcome = .discardedAndRebuilt
            return .discardedAndRebuilt
        }
        restoreLive(&live, confirmed: confirmed)
        events.append("rebuild")
        hostBuildCount += 1
        lastOutcome = .fallbackBuildAfterAccept
        return .fallbackBuildAfterAccept
    }

    private func restoreLive(_ live: inout [TrackGDNMockState], confirmed: Int) {
        guard confirmed > 0 else { return }
        events.append("restore")
        for layer in live.indices {
            live[layer] = bank.restore(layer: layer, position: confirmed - 1)
        }
    }

    private func buildPrebuilt(assumedKeep: Int) {
        events.append("prebuild")
        hostBuildCount += 1
        let startSlot = assumedKeep - 1
        var startState: [TrackGDNMockState] = []
        startState.reserveCapacity(bank.layerCount)
        for layer in 0 ..< bank.layerCount {
            startState.append(bank.restore(layer: layer, position: startSlot))
        }
        prebuilt = TrackPrebuiltDecodeGraph(
            identity: nextIdentity, assumedKeep: assumedKeep,
            startSlot: startSlot, startState: startState, enqueued: false)
        nextIdentity += 1
    }
}
