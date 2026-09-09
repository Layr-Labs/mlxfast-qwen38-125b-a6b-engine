#!/usr/bin/env bash
# Exercise the real setup entrypoint from a plain clone. Only the Swift/Apple
# toolchain is stubbed: git initializes a real local submodule, and curl fetches
# tiny file:// shards through the real parallel downloader and SHA256 verifier.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/setup-onboarding.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
fail() { echo "test-setup-onboarding: FAIL -- $*" >&2; exit 1; }
expect() { grep -Fq -- "$2" "$1" || fail "missing '$2' in $1"; }
reject() { if grep -Fq -- "$2" "$1"; then fail "unexpected '$2' in $1"; fi; }

# Explicit file-only transports make it impossible for a regression to reach a
# remote host. No global git config or developer credentials are used.
export GIT_ALLOW_PROTOCOL=file
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
mkdir -p "${WORK}/engine" "${WORK}/seed/tools" "${WORK}/bin" "${WORK}/upstream"
git -C "${WORK}/engine" init -q
printf '// fixture engine\n' > "${WORK}/engine/Package.swift"
git -C "${WORK}/engine" add Package.swift
git -C "${WORK}/engine" -c user.name=Test -c user.email=test@example.invalid commit -qm 'Fixture engine'
git -C "${WORK}/seed" init -q
git -C "${WORK}/seed" submodule add -q "${WORK}/engine" Vendor/mlx-swift-lm
cp "${ROOT}/setup.sh" "${WORK}/seed/setup.sh"
cp "${ROOT}/tools/stage-bench-worker.sh" "${WORK}/seed/tools/"
printf '// fixture package\n' > "${WORK}/seed/Package.swift"
printf '{}\n' > "${WORK}/seed/Package.resolved"
git -C "${WORK}/seed" add .
git -C "${WORK}/seed" -c user.name=Test -c user.email=test@example.invalid commit -qm 'Fixture setup checkout'

cat > "${WORK}/bin/uname" <<'SH'
#!/usr/bin/env bash
case "$1" in -s) echo Darwin;; -m) echo arm64;; *) exit 1;; esac
SH
cat > "${WORK}/bin/swift" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == build ]] || exit 1
[[ -f Vendor/mlx-swift-lm/Package.swift ]] || { echo 'engine missing before build' >&2; exit 1; }
printf 'build %s\n' "$*" >> "${TEST_EVENTS}"
mkdir -p .build/release .build-worker/release
cp "${TEST_CLI}" .build/release/mlxfast-swift
printf '#!/bin/sh\nexit 0\n' > .build-worker/release/track-bench-worker
chmod +x .build/release/mlxfast-swift .build-worker/release/track-bench-worker
SH
cat > "${WORK}/cli" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  checkpoint-shards)
    printf 'model-01.safetensors\nmodel-02.safetensors\n'
    ;;
  transform)
    [[ "$2" == --reference && "$4" == --output && "$#" == 5 ]] || exit 1
    [[ -x .build/release/bench-worker ]] || { echo 'worker not staged before transform' >&2; exit 1; }
    [[ -f "$3/.mlxfast-reference-cache.lock" ]] || { echo 'reference not verified before transform' >&2; exit 1; }
    printf 'transform %s\n' "$*" >> "${TEST_EVENTS}"
    [[ "${TEST_TRANSFORM_FAIL:-0}" != 1 ]] || { echo 'fixture transform failure' >&2; exit 1; }
    [[ "${TEST_TRANSFORM_EMPTY:-0}" != 1 ]] || exit 0
    mkdir -p "$5"
    printf '{}\n' > "$5/config.json"
    printf '{"weight_map":{}}\n' > "$5/model.safetensors.index.json"
    ;;
  *) echo "unexpected CLI command: $1" >&2; exit 1;;
esac
SH
# Enforce local reads even if setup accidentally ignores the fixture URL.
cat > "${WORK}/bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
[[ "${args[${#args[@]}-1]}" == file://* ]] || { echo 'non-local URL refused by test' >&2; exit 1; }
exec /usr/bin/curl "$@"
SH
chmod +x "${WORK}/bin/"* "${WORK}/cli"
printf '{}\n' > "${WORK}/upstream/config.json"
printf '{"weight_map":{"a":"model-01.safetensors","b":"model-02.safetensors"}}\n' > "${WORK}/upstream/model.safetensors.index.json"
printf 'first shard\n' > "${WORK}/upstream/model-01.safetensors"
printf 'second shard\n' > "${WORK}/upstream/model-02.safetensors"
for path in "${WORK}/upstream/"*; do
  printf '%s %s %s\n' "$(shasum -a 256 "${path}" | awk '{print $1}')" \
    "$(wc -c < "${path}" | tr -d ' ')" "$(basename "${path}")"
done > "${WORK}/reference.sha256"
printf 'fixture metallib\n' > "${WORK}/mlx.metallib"
printf 'fixture fingerprint\n' > "${WORK}/mlx.metallib.fingerprint"

export PATH="${WORK}/bin:${PATH}"
export TEST_CLI="${WORK}/cli"
export MLXFAST_SKIP_MACMON_INSTALL=1 MLXFAST_SKIP_MLX_METALLIB=1
export MLXFAST_MLX_METALLIB="${WORK}/mlx.metallib"
export MLXFAST_REFERENCE_BASE_URL="file://${WORK}/upstream"
export MLXFAST_REFERENCE_FALLBACK_BASE_URL=""
export MLXFAST_REFERENCE_MANIFEST_PATH="${WORK}/reference.sha256"
export MLXFAST_REFERENCE_MIN_FREE_GIB=0 MLXFAST_REFERENCE_DOWNLOAD_JOBS=2
export MLXFAST_REFERENCE_HASH_VERIFY=1 MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY=1
# In particular, do not accidentally inherit the workaround that masked the
# original parallel-download bug in a user's shell.
unset SETUP_LOG_LABEL MLXFAST_SETUP_LOG_LABEL MLXFAST_SKIP_SWIFT_BUILD
unset MLXFAST_SKIP_WEIGHTS_DOWNLOAD SKIP_MODEL_DOWNLOAD

new_clone() {
  CASE_ROOT="${WORK}/$1"
  git clone -q "${WORK}/seed" "${CASE_ROOT}"
  export TEST_EVENTS="${CASE_ROOT}/events"
  export MLXFAST_REFERENCE_DIR="${CASE_ROOT}/reference checkpoint"
  export MLXFAST_WEIGHTS_PATH="${CASE_ROOT}/runtime weights"
}
run_setup() { (cd "${CASE_ROOT}" && bash ./setup.sh) > "${CASE_ROOT}/setup.log" 2>&1; }

new_clone fresh
[[ ! -f "${CASE_ROOT}/Vendor/mlx-swift-lm/Package.swift" ]] || fail 'fixture was not a plain clone'
run_setup || { cat "${CASE_ROOT}/setup.log" >&2; fail 'fresh clone setup failed'; }
expect "${CASE_ROOT}/setup.log" 'initializing the pinned engine submodule'
expect "${CASE_ROOT}/setup.log" 'with 2 parallel job(s)'
expect "${CASE_ROOT}/setup.log" 'setup.sh: downloaded shard 1/2:'
expect "${CASE_ROOT}/setup.log" 'setup.sh: downloaded shard 2/2:'
expect "${CASE_ROOT}/setup.log" 'setup complete'
expect "${CASE_ROOT}/setup.log" "transformed weights: ${MLXFAST_WEIGHTS_PATH}"
[[ -f "${MLXFAST_WEIGHTS_PATH}/config.json" ]] || fail 'runtime weights missing'
[[ "$(tail -1 "${TEST_EVENTS}")" == transform* ]] || fail 'transform did not follow builds'
echo 'test-setup-onboarding: PASS -- plain clone initializes, downloads in parallel, stages, and transforms'

# A cached setup still runs the current transform, and leaves initialized
# dependency edits untouched. A wrapper label must reach shard child shells.
printf '// local edit\n' >> "${CASE_ROOT}/Vendor/mlx-swift-lm/Package.swift"
run_setup || { cat "${CASE_ROOT}/setup.log" >&2; fail 'cached setup failed'; }
reject "${CASE_ROOT}/setup.log" 'initializing the pinned engine submodule'
expect "${CASE_ROOT}/Vendor/mlx-swift-lm/Package.swift" '// local edit'
[[ "$(grep -c '^transform ' "${TEST_EVENTS}")" == 2 ]] || fail 'cached reference skipped current transform'
echo 'test-setup-onboarding: PASS -- cached setup retransforms and preserves submodule edits'

new_clone wrapper
MLXFAST_SETUP_LOG_LABEL=yukon run_setup || { cat "${CASE_ROOT}/setup.log" >&2; fail 'wrapper-labelled setup failed'; }
expect "${CASE_ROOT}/setup.log" 'yukon: downloaded shard 1/2:'
expect "${CASE_ROOT}/setup.log" 'yukon: downloaded shard 2/2:'
echo 'test-setup-onboarding: PASS -- validated wrapper label reaches child shells'

new_clone failed-transform
if TEST_TRANSFORM_FAIL=1 run_setup; then fail 'transform failure was accepted'; fi
expect "${CASE_ROOT}/setup.log" 'fixture transform failure'
reject "${CASE_ROOT}/setup.log" 'setup complete'
echo 'test-setup-onboarding: PASS -- transform failure prevents success summary'

new_clone empty-transform
if TEST_TRANSFORM_EMPTY=1 run_setup; then fail 'empty transform was accepted'; fi
expect "${CASE_ROOT}/setup.log" 'transform did not produce config.json'
reject "${CASE_ROOT}/setup.log" 'setup complete'
echo 'test-setup-onboarding: PASS -- missing transform outputs prevent success summary'

new_clone failed-submodule
git -C "${CASE_ROOT}" config submodule.Vendor/mlx-swift-lm.url "${WORK}/missing-engine"
if run_setup; then fail 'missing submodule remote was accepted'; fi
expect "${CASE_ROOT}/setup.log" 'initializing the pinned engine submodule'
reject "${CASE_ROOT}/setup.log" 'setup complete'
[[ ! -e "${TEST_EVENTS}" ]] || fail 'submodule failure reached the build'
[[ ! -e "${MLXFAST_REFERENCE_DIR}" ]] || fail 'submodule failure started a reference download'
echo 'test-setup-onboarding: PASS -- submodule failure stops before build or download'

new_clone no-weights
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 run_setup || { cat "${CASE_ROOT}/setup.log" >&2; fail 'build-only setup failed'; }
reject "${TEST_EVENTS}" 'transform '
expect "${CASE_ROOT}/setup.log" 'transformed weights: not prepared'
[[ ! -e "${MLXFAST_REFERENCE_DIR}" ]] || fail 'build-only setup downloaded weights'
echo 'test-setup-onboarding: PASS -- explicit build-only setup reports weights as unprepared'
echo 'test-setup-onboarding: all cases pass'
