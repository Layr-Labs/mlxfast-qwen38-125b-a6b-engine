#!/usr/bin/env bash
# Rebuild the editable track worker and Metal library, then stage a checked set.
# This is local provenance for stale-build detection, not a signed attestation
# or a replacement for benchd's runtime identity and correctness checks.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT_DIR}"

fail() { echo "build-bench-worker.sh: $*" >&2; exit 1; }
usage() {
  cat <<'EOF'
Usage: tools/build-bench-worker.sh [--check]

Build the release track-bench-worker product, build mlx.metallib, verify that
the worker links TrackRunner, and stage the set at .build/release/bench-worker.
Run after initial setup or after editing Runner, Swift, or Metal sources.
No model weights are downloaded or loaded.

--check verifies the saved source/toolchain key and staged artifact hashes
without compiling or loading the model. Rebuild when the check reports stale
or missing provenance. Direct stage-bench-worker.sh copies are not certified
as fresh builds and clear the combined command's provenance record.
EOF
}
MODE=build
case "${1:-}" in
  '') ;;
  --check) MODE=check ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 1 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 1; }

# This command intentionally has one product and one layout. Inherited setup
# overrides must not redirect the copy at an old fork product or skip a build.
[[ "${MLXFAST_SWIFT_CONFIGURATION:-release}" == release ]] || fail "this command builds release; unset MLXFAST_SWIFT_CONFIGURATION"
for setting in MLXFAST_SKIP_SWIFT_BUILD MLXFAST_SKIP_MLX_METALLIB; do
  [[ "${!setting:-0}" != 1 ]] || fail "unset ${setting}; this command verifies a complete build"
done
WORKER_REL=.build-worker/release/track-bench-worker
METALLIB_REL=.build-worker/release/mlx.metallib
for setting in MLXFAST_BENCH_WORKER_EXECUTABLE MLXFAST_MLX_METALLIB MLXFAST_MLX_SWIFT_VENDOR; do
  case "${setting}" in
    MLXFAST_BENCH_WORKER_EXECUTABLE) expected="${WORKER_REL}" ;;
    MLXFAST_MLX_METALLIB) expected="${METALLIB_REL}" ;;
    MLXFAST_MLX_SWIFT_VENDOR) expected=Vendor/mlx-swift ;;
  esac
  actual="${!setting:-${expected}}"
  [[ "${actual#./}" == "${expected}" || "${actual}" == "${ROOT_DIR}/${expected}" ]] \
    || fail "${setting} points outside this command's build layout; unset it"
done
export MLXFAST_BENCH_WORKER_EXECUTABLE="${WORKER_REL}"
export MLXFAST_MLX_METALLIB="${METALLIB_REL}"
STAGED_WORKER=.build/release/bench-worker
STAGED_METALLIB=.build/release/mlx.metallib
RECORD="${STAGED_WORKER}.build.json"

for command in git jq shasum swift nm; do
  command -v "${command}" >/dev/null || fail "${command} is required"
done
[[ -f Vendor/mlx-swift-lm/Package.swift ]] \
  || fail "initialize the pinned engine first: git submodule update --init"
pin="$(git ls-tree HEAD Vendor/mlx-swift-lm | awk '{print $3}')"
actual_pin="$(git -C Vendor/mlx-swift-lm rev-parse HEAD)"
[[ -n "${pin}" && "${pin}" == "${actual_pin}" ]] \
  || fail "Vendor/mlx-swift-lm differs from the pinned gitlink; run git submodule update --init"
[[ -z "$(git -C Vendor/mlx-swift-lm status --porcelain -uall)" ]] \
  || fail "Vendor/mlx-swift-lm has local changes; the build requires the pinned engine"

sha256() { shasum -a 256 -- "$1" | awk '{print $1}'; }
source_key() {
  local cache_key extra_sources
  # Reuse the ranked cache's tracked-source, gitlink, Metal, toolchain and root
  # key, adding files SwiftPM can discover without a git add (including ignored
  # files under these source directories) and this command's own recipe.
  cache_key="$(tools/build-cache.sh key)" || return 1
  extra_sources="$(find Runner Sources Plugins Vendor/mlx-swift/Source Vendor/mlx-swift/Plugins \
    \( -type f -o -type l \) -print0 \
    | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')" || return 1
  printf '%s\n' "${cache_key}" "${extra_sources}" "$(sha256 tools/build-bench-worker.sh)" \
    | shasum -a 256 | awk '{print $1}'
}
verify_record() {
  [[ -f "${RECORD}" ]] || fail "no build provenance at ${RECORD}; run tools/build-bench-worker.sh"
  [[ -x "${STAGED_WORKER}" && -f "${STAGED_METALLIB}" && -f "${STAGED_METALLIB}.fingerprint" ]] \
    || fail "the staged worker/Metal set is incomplete; rebuild"
  jq -e --arg source "$(source_key)" --arg worker "$(sha256 "${STAGED_WORKER}")" \
    --arg metal "$(sha256 "${STAGED_METALLIB}")" --arg fingerprint "$(sha256 "${STAGED_METALLIB}.fingerprint")" \
    '.version == 1 and .product == "track-bench-worker" and .source_key == $source
      and .worker_sha256 == $worker and .metallib_sha256 == $metal
      and .metallib_fingerprint_sha256 == $fingerprint' "${RECORD}" >/dev/null \
    || fail "stale or changed worker/Metal set; run tools/build-bench-worker.sh"
  echo "build-bench-worker.sh: source key and staged worker/Metal hashes match ${RECORD}"
}

if [[ "${MODE}" == check ]]; then
  verify_record
  exit 0
fi

before="$(source_key)"
mkdir -p .build-worker/clang-module-cache
# Removing only the expected product forces a relink and prevents a successful
# but misdirected build from reusing a leftover executable at the staged source.
rm -f "${WORKER_REL}"
CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build-worker/clang-module-cache}" \
  swift build -c release --force-resolved-versions --scratch-path .build-worker --product track-bench-worker
[[ -x "${WORKER_REL}" ]] || fail "Swift did not produce ${WORKER_REL}; nothing was staged"
symbols="$(nm -g "${WORKER_REL}")" || fail "cannot inspect the built worker; nothing was staged"
grep -Fq '_$s11TrackRunner' <<< "${symbols}" \
  || fail "built worker does not link this package's editable TrackRunner; nothing was staged"
tools/build-mlx-metallib.sh
expected_fingerprint="$(tools/build-mlx-metallib.sh --print-fingerprint)"
[[ "$(cat "${METALLIB_REL}.fingerprint")" == "mlxfast-metallib-fingerprint-v1 ${expected_fingerprint}" ]] \
  || fail "Metal fingerprint does not match the current vendored sources; nothing was staged"
[[ "$(source_key)" == "${before}" ]] || fail "sources changed during the build; retry before staging"
tools/stage-bench-worker.sh
[[ "$(source_key)" == "${before}" ]] || fail "sources changed during staging; rebuild"
jq -n --arg source "${before}" --arg revision "$(git rev-parse HEAD)" \
  --arg worker "$(sha256 "${STAGED_WORKER}")" --arg metal "$(sha256 "${STAGED_METALLIB}")" \
  --arg fingerprint "$(sha256 "${STAGED_METALLIB}.fingerprint")" \
  '{version:1, product:"track-bench-worker", source_key:$source, checkout_revision:$revision,
    worker_sha256:$worker, metallib_sha256:$metal, metallib_fingerprint_sha256:$fingerprint}' \
  > "${RECORD}.tmp"
mv "${RECORD}.tmp" "${RECORD}"
verify_record
