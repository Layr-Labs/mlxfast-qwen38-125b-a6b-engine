/// Provenance marker for the staged re-measurement candidate.
///
/// This tree carries the editable bytes of public submission
/// `c4370aa2` (a second-measurement resubmission of `f18f2e79`, solver
/// fkiene), overlaid on crown `813953807c935e81f45112167c9736af8894d46c`.
/// Nothing references this declaration; it exists so the staged blob is
/// distinct from the already-scored public blob.
enum TrackRedrawProvenance {
    static let sourceSubmission = "c4370aa2"
    static let overlaidCrown = "813953807c935e81f45112167c9736af8894d46c"
}
