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
//  shard) and the prefill seam (per call).
//

import Foundation

enum TrackBootPhase {
    /// Mark a boot phase. Unbuffered, so the line survives a hard death that
    /// follows it. Callers that fire more than once (per-shard progress, the
    /// per-prefill seam marker) print per call — one line each, no state.
    static func mark(_ label: String) {
        fputs("bench-worker: track-boot: \(label)\n", stderr)
        fflush(stderr)
    }
}
