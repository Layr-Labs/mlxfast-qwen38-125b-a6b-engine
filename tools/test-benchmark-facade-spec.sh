#!/usr/bin/env bash
# Drive the real facade with a recording benchd: declaration-to-wire parity,
# refusal before a run, and honest pre-run diagnostics. No model or GPU loads.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
ROOT="${WORK}/track"
mkdir -p "${ROOT}/tools" "${ROOT}/fixtures"
cp "${REPO_ROOT}/tools/benchmark.sh" "${REPO_ROOT}/tools/spec-declaration.sh" "${ROOT}/tools/"
cp "${REPO_ROOT}/benchmark.json" "${ROOT}/"
cp "${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json" "${ROOT}/fixtures/"
printf '{}\n' > "${ROOT}/golden.json"

cat > "${WORK}/benchd" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == 'iterate --help' ]]; then
  printf '%s\n' '--engine-resource <k=v>'
  if [[ "${STUB_MTP_SUPPORTED}" == '1' ]]; then printf '%s\n' '--mtp-depth <N>'; fi
  exit 0
fi
printf '%s\n' "$@" > "${STUB_ARGV}"
printf '%s\n' '{"stub":"sealed payload passes through unchanged"}'
exit "${STUB_EXIT_CODE}"
STUB
chmod +x "${WORK}/benchd"
# Deliberately not a worker: the facade must never invoke an engine to get
# pre-run identity. It can hash the bytes without loading a model.
printf 'do not execute this worker\n' > "${WORK}/worker with spaces"
worker_sha="$(shasum -a 256 "${WORK}/worker with spaces" | awk '{print $1}')"

failures=0
checks=0
check() {
  checks=$((checks + 1))
  if ! "$@"; then
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
  fi
}

run_facade() {
  local mode="$1"
  rm -f "${WORK}/argv"
  rc=0
  env -u MLXFAST_QWEN_MTP_TRACK_ID -u SPEC_DECLARATION_MANIFEST \
    -u SPEC_DECLARATION_CONTRACT -u BENCH_WORKER_RESIDENT_SOCKET \
    -u MLXFAST_CORRECTNESS_GOLDEN_SHA256 -u MLXFAST_CORRECTNESS_GOLDEN_BYTES \
    MLXFAST_USE_RUNTIME_WORKER=1 MLXFAST_NO_SANDBOX=0 \
    BENCHD="${WORK}/benchd" MLXFAST_ENGINE_BIN="${WORK}/worker with spaces" \
    MLXFAST_CORRECTNESS_GOLDEN_PATH="${ROOT}/golden.json" \
    MLXFAST_SCORE_PATH="${WORK}/score.json" STUB_ARGV="${WORK}/argv" \
    STUB_MTP_SUPPORTED="${supported}" STUB_EXIT_CODE="${stub_rc}" \
    "${ROOT}/tools/benchmark.sh" "--${mode}" > "${WORK}/stdout" 2> "${WORK}/stderr" || rc=$?
}

arg_equals() {
  [[ -f "${WORK}/argv" ]] && [[ "$(awk -v flag="$1" '$0 == flag {getline; print}' "${WORK}/argv")" == "$2" ]]
}
has_arg() { [[ -f "${WORK}/argv" ]] && grep -Fxq -- "$1" "${WORK}/argv"; }
stderr_has() { grep -Fq -- "$1" "${WORK}/stderr"; }
stdout_untouched() { [[ "$(cat "${WORK}/stdout")" == '{"stub":"sealed payload passes through unchanged"}' ]]; }
refused_before_run() { [[ "${rc}" -eq 1 && ! -f "${WORK}/argv" ]]; }

supported=1
stub_rc=0
for depth in 1 2 3 4 5 6; do
  printf '{"spec":{"enabled":true,"num_speculative_tokens":%s}}\n' "${depth}" > "${ROOT}/mtp-head.manifest.json"
  for mode in local-iterate local-submit official; do
    run_facade "${mode}"
    check test "${rc}" -eq 0
    check arg_equals --mtp-depth "${depth}"
    check arg_equals --mode "${mode}"
    check arg_equals --engine "${WORK}/worker with spaces"
    check stdout_untouched
    check stderr_has "mode=${mode}, decode=mtp, mtp_depth=${depth}, batch_size=1"
    check stderr_has "worker_sha256=${worker_sha}"
    if [[ "${mode}" == official ]]; then
      check arg_equals --contract "${ROOT}/tools/../fixtures/qwen3_8_125b_a6b_track.json"
      check test "$(grep -cFx -- '--cool-gate' "${WORK}/argv" || true)" -eq 0
    else
      check has_arg --cool-gate
    fi
  done
done

# Absent, disabled and zero-depth declarations are serial in every mode,
# including with an older benchd that has no speculative request support.
supported=0
for declaration in absent '{}' '{"spec":{"enabled":false,"num_speculative_tokens":3}}' '{"spec":{"enabled":true,"num_speculative_tokens":0}}'; do
  if [[ "${declaration}" == absent ]]; then
    rm -f "${ROOT}/mtp-head.manifest.json"
  else
    printf '%s\n' "${declaration}" > "${ROOT}/mtp-head.manifest.json"
  fi
  for mode in local-iterate local-submit official; do
    run_facade "${mode}"
    check test "${rc}" -eq 0
    check test "$(grep -cFx -- '--mtp-depth' "${WORK}/argv" || true)" -eq 0
    check stderr_has "mode=${mode}, decode=serial, mtp_depth=0, batch_size=1"
  done
done

printf '{"spec":{"enabled":true,"num_speculative_tokens":3}}\n' > "${ROOT}/mtp-head.manifest.json"
for mode in local-iterate local-submit official; do
  run_facade "${mode}"
  check refused_before_run
  check stderr_has 'requests MTP depth 3 but this benchd has no --mtp-depth'
done

supported=1
for declaration in '{broken' '{"spec":{"enabled":true,"num_speculative_tokens":7}}' '{"spec":{"enabled":"yes","num_speculative_tokens":3}}'; do
  printf '%s\n' "${declaration}" > "${ROOT}/mtp-head.manifest.json"
  run_facade local-iterate
  check refused_before_run
  check stderr_has 'spec-declaration.sh: REFUSING'
done

# No invalid batch size can be labeled as an actual single-stream selection.
rm "${ROOT}/mtp-head.manifest.json"
contract="${ROOT}/fixtures/qwen3_8_125b_a6b_track.json"
jq '.scored_batch_size = 2' "${contract}" > "${WORK}/changed-contract"
cp "${WORK}/changed-contract" "${contract}"
run_facade local-iterate
check refused_before_run
check stderr_has 'requires scored_batch_size=1'
cp "${REPO_ROOT}/fixtures/qwen3_8_125b_a6b_track.json" "${contract}"

# Diagnostics never convert benchd failure into success; keep the facade's
# existing usage-exit mapping. An unversioned checkout is labeled as such.
for stub_rc in 1 2; do
  run_facade local-iterate
  check test "${rc}" -eq 1
  check stderr_has 'checkout_revision=unavailable'
done

if [[ "${failures}" -ne 0 ]]; then
  echo "FAILED: ${failures} of ${checks} facade spec checks" >&2
  exit 1
fi
echo "OK: ${checks} facade spec checks passed (no model or GPU)"
