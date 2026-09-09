#!/usr/bin/env bash
# Real build/stage/provenance scripts and cache key, with compiler/Metal stubs.
# No Swift build, model, GPU, or network. Proves which product is requested,
# freshness checks, and refusal without replacing a usable staged set.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
ROOT="${WORK}/checkout with spaces"
mkdir -p "${ROOT}/tools" "${ROOT}/Runner" "${ROOT}/Sources" "${ROOT}/Plugins" \
  "${ROOT}/Vendor/mlx-swift-lm" "${ROOT}/Vendor/mlx-swift/Source/MLX" \
  "${ROOT}/Vendor/mlx-swift/Plugins" "${WORK}/bin"
cp "${REPO_ROOT}/tools/"{build-bench-worker,stage-bench-worker,build-cache}.sh "${ROOT}/tools/"
printf 'editable runner\n' > "${ROOT}/Runner/Model.swift"
printf 'fixed package\n' > "${ROOT}/Package.swift"
printf '{}\n' > "${ROOT}/Package.resolved"
printf 'engine package\n' > "${ROOT}/Vendor/mlx-swift-lm/Package.swift"

cat > "${ROOT}/tools/build-mlx-metallib.sh" <<'METAL'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --print-fingerprint ]]; then printf '%064d\n' 1; exit 0; fi
[[ "${BUILD_CASE:-}" != metal-fails ]] || exit 1
printf 'new metallib\n' > .build-worker/release/mlx.metallib
printf 'mlxfast-metallib-fingerprint-v1 %064d\n' 1 > .build-worker/release/mlx.metallib.fingerprint
if [[ "${BUILD_CASE:-}" == source-race ]]; then printf 'mid-build change\n' >> Runner/Model.swift; fi
if [[ "${BUILD_CASE:-}" == bad-fingerprint ]]; then printf 'wrong\n' > .build-worker/release/mlx.metallib.fingerprint; fi
METAL
cat > "${WORK}/bin/swift" <<'SWIFT'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --version ]]; then echo 'stub Swift 6.3'; exit 0; fi
printf '%s\n' "$@" > "${BUILD_LOG}"
[[ "$*" == 'build -c release --force-resolved-versions --scratch-path .build-worker --product track-bench-worker' ]] || exit 2
[[ "${BUILD_CASE:-}" != swift-fails ]] || exit 1
[[ "${BUILD_CASE:-}" != no-output ]] || exit 0
mkdir -p .build-worker/release
if [[ "${BUILD_CASE:-}" == wrong-worker ]]; then
  printf '#!/bin/sh\n# fork product\nexit 0\n' > .build-worker/release/track-bench-worker
else
  printf '#!/bin/sh\n# TrackRunner product\nexit 0\n' > .build-worker/release/track-bench-worker
fi
chmod +x .build-worker/release/track-bench-worker
SWIFT
cat > "${WORK}/bin/nm" <<'NM'
#!/usr/bin/env bash
set -euo pipefail
if grep -q TrackRunner "$2"; then printf '%s\n' '0000 S _$s11TrackRunner0a8Qwen4ExpB0C'; fi
NM
chmod +x "${ROOT}/tools/"*.sh "${WORK}/bin/"*
git -C "${ROOT}/Vendor/mlx-swift-lm" init -q
git -C "${ROOT}/Vendor/mlx-swift-lm" add Package.swift
git -C "${ROOT}/Vendor/mlx-swift-lm" -c user.name=Test -c user.email=test@example.invalid commit -qm 'pinned engine'
pin="$(git -C "${ROOT}/Vendor/mlx-swift-lm" rev-parse HEAD)"
git -C "${ROOT}" init -q
git -C "${ROOT}" add tools Runner Package.swift Package.resolved
git -C "${ROOT}" update-index --add --cacheinfo "160000,${pin},Vendor/mlx-swift-lm"
printf '.build*/\n' > "${ROOT}/.gitignore"
git -C "${ROOT}" add .gitignore
git -C "${ROOT}" -c user.name=Test -c user.email=test@example.invalid commit -qm 'synthetic engine'

run_build() {
  rc=0
  env -u MLXFAST_BENCH_WORKER_EXECUTABLE -u MLXFAST_MLX_METALLIB \
    -u MLXFAST_SWIFT_CONFIGURATION -u MLXFAST_MLX_SWIFT_VENDOR \
    -u MLXFAST_SKIP_SWIFT_BUILD -u MLXFAST_SKIP_MLX_METALLIB \
    PATH="${WORK}/bin:${PATH}" BUILD_LOG="${WORK}/build.argv" "$@" \
    > "${WORK}/stdout" 2> "${WORK}/stderr" || rc=$?
}
BUILD="${ROOT}/tools/build-bench-worker.sh"
STAGED="${ROOT}/.build/release/bench-worker"
checks=0
check() { checks=$((checks + 1)); if ! "$@"; then cat "${WORK}/stderr" >&2; echo "FAIL: $*" >&2; exit 1; fi; }
stderr_has() { grep -Fq -- "$1" "${WORK}/stderr"; }

# A fork binary left by the old docs must never become the source product.
mkdir -p "${ROOT}/.build-worker/release"
printf '#!/bin/sh\n# stale fork\nexit 0\n' > "${ROOT}/.build-worker/release/bench-worker"
chmod +x "${ROOT}/.build-worker/release/bench-worker"
run_build "${BUILD}"
check test "${rc}" -eq 0
check grep -Fq TrackRunner "${STAGED}"
check cmp "${ROOT}/.build-worker/release/track-bench-worker" "${STAGED}"
check test -f "${STAGED}.build.json"
check grep -Fxq -- '--force-resolved-versions' "${WORK}/build.argv"
check grep -Fxq -- 'track-bench-worker' "${WORK}/build.argv"
run_build "${BUILD}" --check
check test "${rc}" -eq 0

# Tracked and untracked edits both invalidate provenance without compiling.
printf 'edited\n' >> "${ROOT}/Runner/Model.swift"
run_build "${BUILD}" --check
check test "${rc}" -eq 1
check stderr_has 'stale or changed'
git -C "${ROOT}" checkout -- Runner/Model.swift
printf 'new source\n' > "${ROOT}/Runner/Untracked.swift"
run_build "${BUILD}" --check
check test "${rc}" -eq 1
rm "${ROOT}/Runner/Untracked.swift"

# SwiftPM also discovers files in the vendored MLX targets. Neither the
# tracked cache key nor Cmlx's Metal fingerprint covers a new MLX Swift file.
printf 'new vendored source\n' > "${ROOT}/Vendor/mlx-swift/Source/MLX/Added.swift"
run_build "${BUILD}" --check
check test "${rc}" -eq 1
rm "${ROOT}/Vendor/mlx-swift/Source/MLX/Added.swift"
printf 'Vendor/mlx-swift/Source/MLX/Ignored.swift\n' >> "${ROOT}/.git/info/exclude"
printf 'ignored source still compiles\n' > "${ROOT}/Vendor/mlx-swift/Source/MLX/Ignored.swift"
check git -C "${ROOT}" check-ignore -q Vendor/mlx-swift/Source/MLX/Ignored.swift
run_build "${BUILD}" --check
check test "${rc}" -eq 1
rm "${ROOT}/Vendor/mlx-swift/Source/MLX/Ignored.swift"

cp "${STAGED}" "${WORK}/saved-worker"
printf 'tampered\n' >> "${STAGED}"
run_build "${BUILD}" --check
check test "${rc}" -eq 1
cp "${WORK}/saved-worker" "${STAGED}"
cp "${ROOT}/.build/release/mlx.metallib" "${WORK}/saved-metal"
printf 'tampered\n' >> "${ROOT}/.build/release/mlx.metallib"
run_build "${BUILD}" --check
check test "${rc}" -eq 1
cp "${WORK}/saved-metal" "${ROOT}/.build/release/mlx.metallib"

# Neither failed builds nor a stale leftover count as a newly built product.
for build_case in swift-fails no-output wrong-worker metal-fails bad-fingerprint source-race; do
  run_build BUILD_CASE="${build_case}" "${BUILD}"
  check test "${rc}" -ne 0
  check cmp "${WORK}/saved-worker" "${STAGED}"
  check cmp "${WORK}/saved-metal" "${ROOT}/.build/release/mlx.metallib"
done
git -C "${ROOT}" checkout -- Runner/Model.swift

run_build MLXFAST_BENCH_WORKER_EXECUTABLE=.build-worker/release/bench-worker "${BUILD}"
check test "${rc}" -eq 1
check stderr_has 'MLXFAST_BENCH_WORKER_EXECUTABLE points outside'
run_build MLXFAST_SKIP_SWIFT_BUILD=1 "${BUILD}"
check test "${rc}" -eq 1
check stderr_has 'unset MLXFAST_SKIP_SWIFT_BUILD'
printf 'dirty engine\n' >> "${ROOT}/Vendor/mlx-swift-lm/Package.swift"
run_build "${BUILD}"
check test "${rc}" -eq 1
check stderr_has 'Vendor/mlx-swift-lm has local changes'
git -C "${ROOT}/Vendor/mlx-swift-lm" checkout -- Package.swift

# Successful direct staging clears the provenance it cannot vouch for.
run_build "${BUILD}"
check test "${rc}" -eq 0
run_build "${ROOT}/tools/stage-bench-worker.sh"
check test "${rc}" -eq 0
check test ! -e "${STAGED}.build.json"
run_build "${BUILD}" --check
check test "${rc}" -eq 1
check stderr_has 'no build provenance'

echo "test-build-bench-worker.sh: ${checks} checks passed (no compiler or GPU)"
