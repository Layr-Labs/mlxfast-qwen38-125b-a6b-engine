// Closed-set Metal template keys for per-call host scalars.
//
// GDN recurrence length T and RMS eps were encoded on every launch. Known
// production values become template / dispatch-key constants; anything else
// keeps the original scalar argument. `TRACK_SCALAR_TEMPLATES=0` restores
// every original dispatch site.

import Foundation

public enum TrackScalarTemplates {
    /// Default ON. `TRACK_SCALAR_TEMPLATES=0` restores scalar T / eps arguments.
    nonisolated(unsafe) public static var enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_SCALAR_TEMPLATES"] ?? "1") != "0"
    }()

    /// Tests only: force the generic (scalar-argument) kernels.
    nonisolated(unsafe) static var forceGeneric = false

    /// Decode T=1, MTP windows 2...8, and the scored 1024-token prefill plus
    /// the power-of-two lengths the engine actually uses for those windows.
    static func isKnownT(_ t: Int) -> Bool {
        switch t {
        case 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024:
            return true
        default:
            return false
        }
    }

    /// Model config `rms_norm_eps` (and the gated-RMS hard 1e-6).
    static let eps1e6Bits = Int(Float(1e-6).bitPattern)

    static func isKnownEps(_ eps: Float) -> Bool {
        Int(eps.bitPattern) == eps1e6Bits
    }

    static func useTemplateT(_ t: Int) -> Bool {
        enabled && !forceGeneric && isKnownT(t)
    }

    static func useTemplateEps(_ eps: Float) -> Bool {
        enabled && !forceGeneric && isKnownEps(eps)
    }
}
