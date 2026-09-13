// TrackPLEVerifyPrefetch.swift -- overlap MTP-verify PLE gather with layer-0 GDN.
//
// Serial/chained ids are host-created; asArray at fastStreams entry is a
// memcpy (w05). MTP verify windows concatenate lazy GPU draft-token ids, so
// that same asArray waits for every draft head and then hashes and gathers
// with the GPU idle. This path copies those ids asynchronously, submits the
// layer-0 GDN graph, and only then reads the ids and gathers the 16 PLE rows.
//
// TRACK_PLE_VERIFY_PREFETCH=0 restores the w05 entry read. An unexpected
// seam (PLE not at layer 1, layer 0 not GDN) also keeps w05.

import Foundation

enum TrackPLEVerifyPrefetch {
    /// Kill switch. Unset or any value other than "0" is on.
    static let enabled: Bool = {
        ProcessInfo.processInfo.environment["TRACK_PLE_VERIFY_PREFETCH"] != "0"
    }()

    /// Layer 0 is GDN without PLE; PLE is at layer 1. Anything else is not
    /// this overlap.
    static func seamOK(layer0IsGDN: Bool, layer0HasPLE: Bool, pleLayerIndex: Int?) -> Bool {
        layer0IsGDN && !layer0HasPLE && pleLayerIndex == 1
    }

    /// Capture-verify window of 2...8 tokens on the expected seam.
    static func shouldDelay(
        window: Int, capture: Bool, seamOK: Bool,
        enabled: Bool = TrackPLEVerifyPrefetch.enabled
    ) -> Bool {
        enabled && capture && seamOK && window > 1 && window <= 8
    }

    /// Copy the lazy ids, submit layer-0, then asArray + hash + gather.
    static func run(
        copyIds: () -> Void,
        buildAndSubmitLayer0: () -> Void,
        readIdsAndGather: () -> Void
    ) {
        copyIds()
        buildAndSubmitLayer0()
        readIdsAndGather()
    }
}
