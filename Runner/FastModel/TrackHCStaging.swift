// Compile-time staging of hyper-connection walks (PB-214 / PB-507).
//
// Presence check (w66):
//   * injectNorm / injectNormWide: no serial k-loop. N_READS is a source
//     literal 4; HC is grid.y; simd_groups is constexpr from template H.
//     Nothing to stage. Threadgroup footprint unchanged.
//   * track_inject_qmv / track_inject_qmv_row: in_vec_size is already a
//     Metal template int (NFULL is constexpr). Staging interpolates the live
//     K into the instantiation so the bound is a source literal, not only a
//     template parameter (c0e42c7e "Scope of the constant").
//   * mixer downInject qmv_fast_reg / qmv_wide_*: in_vec_size was a runtime
//     function argument. Dual-inject body. Staged.
//   * mixer upMix qmv_reg_rows / HC combine: runtime K and template HC.
//     PB-507 tail. Staged.
//   * hcMix `for (s = 0; s < HC; ++s)`: HC was a kernel template int, not a
//     source literal. PB-507 tail. Staged.
//
// Staging lives in registers. No new threadgroup arrays. Existing
// `fp[8 * VPT]` and `res[8][VPT]` keep their sizes for a given window.
//
// TRACK_HC_STAGING defaults ON. `0` restores the generic kernels.

import Foundation

enum TrackHCStaging {
    static var enabled: Bool {
        (ProcessInfo.processInfo.environment["TRACK_HC_STAGING"] ?? "1") != "0"
    }

    /// Scored-model hyper-connection width.
    static let liveHC = 4
    /// Scored-model hidden (inject / mixer-down K).
    static let liveHidden = 2560
    /// Scored-model mixer low-rank (up K).
    static let liveLowrank = 320
    /// Depth-0 serial decode is S=1. MTP verify widths 1...7 exist behind spec flags.
    static let liveWindows: [Int] = [1, 2, 3, 4, 5, 6, 7]

    enum Path: Equatable { case staged, generic }

    static func mixerPath(s: Int, kd: Int, hc: Int, staging: Bool = enabled) -> Path {
        if staging, liveWindows.contains(s), kd == liveHidden, hc == liveHC { return .staged }
        return .generic
    }

    static func upMixPath(s: Int, lw: Int, hc: Int, staging: Bool = enabled) -> Path {
        if staging, liveWindows.contains(s), lw == liveLowrank, hc == liveHC { return .staged }
        return .generic
    }

    static func mixPath(hc: Int, staging: Bool = enabled) -> Path {
        if staging, hc == liveHC { return .staged }
        return .generic
    }

    static func mixerHeadPath(kdim: Int, hc: Int, staging: Bool = enabled) -> Path {
        if staging, kdim == liveHidden, hc == liveHC { return .staged }
        return .generic
    }
}
