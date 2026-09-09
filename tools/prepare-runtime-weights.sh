#!/usr/bin/env bash
# Prepare fresh output before replacing any existing local runtime weights.
set -euo pipefail
if [[ "$#" != 3 ]]; then
  echo 'usage: prepare-runtime-weights.sh CLI REFERENCE OUTPUT' >&2
  exit 2
fi
CLI="$1"
REFERENCE="$2"
OUTPUT="$3"
fail() { echo "prepare-runtime-weights.sh: $*" >&2; exit 1; }
while [[ "${OUTPUT}" == */ && "${OUTPUT}" != / ]]; do OUTPUT="${OUTPUT%/}"; done

# The transform validates its temporary output, so validate the eventual
# destination too. Never replace a symlink, the checkout, or the reference.
name="$(basename "${OUTPUT}")"
[[ -n "${OUTPUT}" && "${name}" != . && "${name}" != .. && "${name}" != / ]] \
  || fail 'output must name a weights directory'
[[ ! -L "${OUTPUT}" ]] || fail "output is a symlink; use its intended directory explicitly: ${OUTPUT}"
[[ ! -e "${OUTPUT}" || -d "${OUTPUT}" ]] || fail "output exists and is not a directory: ${OUTPUT}"
mkdir -p "$(dirname "${OUTPUT}")"
parent="$(cd -P "$(dirname "${OUTPUT}")" && pwd -P)"
OUTPUT="${parent%/}/${name}"
reference="$(cd -P "${REFERENCE}" && pwd -P)"
checkout="$(pwd -P)"
[[ "${OUTPUT}" != "${reference}" && "${reference}" != "${OUTPUT}/"* \
    && "${OUTPUT}" != "${reference}/"* ]] || fail 'output and reference directories must be separate'
[[ "${checkout}" != "${OUTPUT}" && "${checkout}" != "${OUTPUT}/"* ]] \
  || fail 'output must not contain the current working directory'

lock="${OUTPUT}.setup-lock"
mkdir "${lock}" 2>/dev/null || fail "output is already being prepared, or a previous setup left a lock: ${lock}"
work=""
restore_previous=0
cleanup() {
  local status="$?"
  trap - EXIT
  if [[ "${restore_previous}" == 1 && -d "${work}/previous" ]]; then
    if [[ ! -e "${OUTPUT}" && ! -L "${OUTPUT}" ]]; then
      if ! mv "${work}/previous" "${OUTPUT}"; then
        echo "prepare-runtime-weights.sh: restore failed; previous weights remain at ${work}/previous" >&2
        work=""
        status=1
      fi
    else
      echo "prepare-runtime-weights.sh: previous weights retained at ${work}/previous after interrupted publication" >&2
      work=""
      status=1
    fi
  fi
  [[ -z "${work}" ]] || rm -rf "${work}"
  rmdir "${lock}" || status=1
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
work="$(mktemp -d "${parent}/.${name}.setup.XXXXXX")"
prepared="${work}/prepared"
"${CLI}" transform --reference "${REFERENCE}" --output "${prepared}"
[[ -d "${prepared}" && ! -L "${prepared}" \
    && -f "${prepared}/config.json" && ! -L "${prepared}/config.json" \
    && -f "${prepared}/model.safetensors.index.json" && ! -L "${prepared}/model.safetensors.index.json" ]] \
  || fail 'transform did not produce config.json and model.safetensors.index.json in a fresh output tree'

# Same-filesystem renames publish the completed tree. On platforms without a
# directory-exchange primitive there is a short gap between these two renames;
# callers must not load weights concurrently with setup. Roll back the first
# rename if publication fails, keeping prior weights until the new tree is ready.
if [[ -e "${OUTPUT}" ]]; then
  restore_previous=1
  mv "${OUTPUT}" "${work}/previous"
fi
mv "${prepared}" "${OUTPUT}"
restore_previous=0
echo "prepare-runtime-weights.sh: runtime weights prepared at ${OUTPUT}"
