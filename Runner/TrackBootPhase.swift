//
//  TrackBootPhase.swift
//
//  One unbuffered stderr line per boot phase, so a resident that dies on the
//  ranked box leaves a located trail in the log `tools/resident-up.sh` tails
//  on failure. Submission 6 (ef0754d1) exited silently, by code, with an
//  empty log ~30 s into its boot — every failure path in the worker prints,
//  so the only way the next such death stays diagnosable is a marker at each
//  seam it can die between. The writes are unconditional and carry no state;
//  each call site fires once per boot, except the progress markers (per
//  shard), the prefill seam (per call), and the forward seam (per call).
//
//  Submission 7 (f7211c6b) proved the load completes and the death lands in
//  the FIRST candidate-specific code after it: the resident warm pass
//  (`BenchWorkerResidentWarm`, on by default), whose stepper drives
//  `cbv2Forward`. The warm seams below bracket that window from the editable
//  side — the pass itself is fork code and cannot carry markers.
//

import Foundation

enum TrackBootPhase {
    /// Marks emitted at decode width (S == 1) before the seam goes quiet.
    /// The warm pass emits 8; the budget leaves margin and then stops, so
    /// the marker never sits inside a timed teacher-forced decode loop
    /// (`decode_step` reaches the same forward once per measured token).
    static let decodeWidthBudget = 16
    nonisolated(unsafe) private static var decodeWidthMarks = 0

    /// Mark a boot phase. Unbuffered, so the line survives a hard death that
    /// follows it. Callers that fire more than once (per-shard progress, the
    /// per-prefill seam marker) print per call — one line each, no state.
    /// The forward seam at decode width is the one exception: it prints its
    /// first `decodeWidthBudget` lines, then stops, because that seam fires
    /// once per token in steady state.
    static func mark(_ label: String) {
        fputs("bench-worker: track-boot: \(label)\n", stderr)
        fflush(stderr)
    }

    static func markForwardBegin(width: Int) {
        if width == 1 {
            decodeWidthMarks += 1
            guard decodeWidthMarks <= decodeWidthBudget else { return }
        }
        mark("cbv2 forward begin (S=\(width))")
    }
}
