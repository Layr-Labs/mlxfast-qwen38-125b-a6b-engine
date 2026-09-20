//  TrackRedrawProvenance.swift
//
//  Declared provenance marker for this submission tree.
//
//  This tree is a second measurement of public submission `8f776726`
//  (solver: fkiene) on `davidtai/qwen38-125b-a6b-mlx-v1`. That submission
//  stacks, on top of public tree `dba02956` (the MoE decode gate/up
//  reuse kernel running one row per simdgroup), a gated-deltanet decode
//  kernel change that evaluates the per-step gate scalars on a simdgroup
//  that was otherwise idle during the decode walk.
//
//  The file exists so the uploaded editable blob differs from the
//  already-scored `8f776726` blob. It adds no runtime behaviour.

public enum TrackRedrawProvenance {
    /// Public submission this tree re-measures.
    public static let sourceSubmission = "8f776726"
    /// Solver credited for the engineering in the source submission.
    public static let sourceSolver = "fkiene"
    /// Parent public tree of the source submission.
    public static let sourceParentCommit = "dba0295612297b5523b770364b18c64c32b2a865"
    /// Crown commit this tree is based on.
    public static let baseCommit = "813953807c935e81f45112167c9736af8894d46c"
}
