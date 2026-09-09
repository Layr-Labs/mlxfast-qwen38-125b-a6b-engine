#!/usr/bin/env bash
# Run the public local fixture through the normal benchmark entry point.
# benchd continues to own correctness, cooling, timing, and sealed artifacts.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/local-baseline.sh

Run the public Qwen MLX local baseline after setup and weight transformation.
No organizer goldens, reference workspace, or calibration file are required.
This measures local correctness and timing; it does not produce a ranked score.

Optional environment overrides (relative paths resolve from the checkout):
  MLXFAST_ENGINE_BIN                .build/release/bench-worker
  MLXFAST_CORRECTNESS_GOLDEN_PATH    correctness_prompts/public_longcopy_gate_english_1024_256.json
  MLXFAST_WEIGHTS_PATH              weights
  MLXFAST_SCORE_PATH                score.local-iterate.json

The existing benchmark entry point verifies benchd and enables the cool gate.
EOF
}

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_ROOT}"

export MLXFAST_ENGINE_BIN="${MLXFAST_ENGINE_BIN:-.build/release/bench-worker}"
export MLXFAST_CORRECTNESS_GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_prompts/public_longcopy_gate_english_1024_256.json}"
export MLXFAST_WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-weights}"
export MLXFAST_SCORE_PATH="${MLXFAST_SCORE_PATH:-score.local-iterate.json}"

# This helper runs one candidate against the public fixture. benchd also reads
# paired-run settings from the environment, so an operator's shell must not
# accidentally turn this local entry point into a reference/candidate pair.
unset MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION

if [[ ! -x "${MLXFAST_ENGINE_BIN}" ]]; then
  echo "local-baseline.sh: worker not executable: ${MLXFAST_ENGINE_BIN}" >&2
  echo "local-baseline.sh: run ./setup.sh to build and stage the worker, or set MLXFAST_ENGINE_BIN" >&2
  exit 1
fi

{
  echo "local-baseline.sh: public local baseline (unranked)"
  echo "local-baseline.sh: golden=${MLXFAST_CORRECTNESS_GOLDEN_PATH}"
  echo "local-baseline.sh: result=${MLXFAST_SCORE_PATH}; score=null is expected without ranked paired scoring"
} >&2

exec "${REPO_ROOT}/benchmark.sh" --local-iterate
