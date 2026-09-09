#!/usr/bin/env bash
# resident-up.sh -- boot a resident bench-worker, either FOR ONE LEG (benchd
# drives it: --boot / --stop) or around one command (the local wrapper form).
#
# TWO FORMS, AND THE RANKED PATH USES THE FIRST.
#
#   tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N> \
#                        --socket-out <file>
#   tools/resident-up.sh --stop --socket <path>
#
#     The PER-LEG form. benchd calls it once per leg, and it calls THE COPY IN
#     THAT LEG'S OWN TREE. The boot loads that tree's own engine and that tree's
#     own weights, waits for a healthy hello, writes the socket path and exits 0
#     with the resident STILL RUNNING. benchd then measures the leg and calls
#     --stop. This is the Mac counterpart of the CUDA track's serve-up
#     convention, with the same argv.
#
#     WHY THIS EXISTS (public run 34230122059). The ranked measure script used
#     to boot ONE resident from the CANDIDATE tree and export
#     BENCH_WORKER_RESIDENT_SOCKET into benchd. Under the paired design the
#     reference leg then attached to the candidate's resident and was refused:
#     "resident holds <candidate>/weights but this phase asked for
#     <baseline-workspace>/weights". One resident per WINDOW is wrong when a
#     window has two legs on two trees. One resident per LEG is right, and only
#     benchd knows where a leg's boundaries are -- so benchd boots them.
#
#   tools/resident-up.sh --weights <dir> [...] -- <command> [args...]
#
#     The WRAPPER form, kept for the LOCAL UNSCORED path. It has no caller left
#     in this repository after the ranked path stopped wrapping: its remaining
#     users are a participant's hand run and anything that wants
#     BENCH_WORKER_RESIDENT_SOCKET exported around a command (tools/benchmark.sh
#     honours that variable). It is unchanged.
#
# WEIGHTS LOAD ONCE PER WINDOW. benchd spawns `bench-worker runtime-worker`
# once per phase (warmup, timed prefill, timed decode, correctness, and again
# for every leg). In process, each spawn loads the 113 GB checkpoint. So the
# weights get an OWNER: `bench-worker resident` (fork contract section 12f).
# This script boots exactly one of them, inside the caller's GPU-lock window,
# and exports BENCH_WORKER_RESIDENT_SOCKET. Every per-phase worker attaches to
# it and loads nothing. This is the Mac counterpart of the CUDA track's
# tools/serve-up.sh, with the same rules: caller owns the lock, one resident,
# no path leaves it up.
#
# GPU LOCK -- NOT TAKEN HERE, BUT REQUIRED. The caller owns the box GPU window
# (the ranked workflow holds /tmp/mtplx-gpu-exclusive.lock for the whole
# measurement) and this script boots the resident inside it. A resident holds
# ~113 GB of unified memory, so it must never outlive that window. The script
# checks that the lock is HELD (a non-blocking try-lock must fail) and refuses
# by name when nobody holds it. It never takes the lock itself, because a
# lock this script took would be released when this script exits, which is
# not the window.
#
# ONE RESIDENT PER BOX. A second resident would double-load the checkpoint.
# The script refuses by name when its pidfile names a live resident, or when a
# resident already answers on the socket.
#
# Usage:
#   tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N>
#                        --socket-out <file>
#   tools/resident-up.sh --stop --socket <path>
#   tools/resident-up.sh --weights <dir> [--ngram <dir>] [--hello-identity]
#                        [--socket <path>] -- <command> [args...]
#
# --boot:
#   --spec serial|mtp  what this leg is. THE FLAG IS AUTHORITATIVE. Nothing in
#                      the boot path reads the tree's mtp-head.manifest.json or
#                      tools/spec-declaration.sh, so a reference tree whose
#                      manifest declares a draft depth still boots a SERIAL
#                      control leg when benchd says serial. Deriving the value
#                      from the tree is exactly the leak this flag closes.
#   --draft-len <N>    0 with `serial`; 1 to 6 with `mtp`. A mismatch refuses.
#   --socket-out <f>   the boot writes the resident's socket path as the FIRST
#                      LINE of this file. A `<socket>.pid` sidecar and a
#                      `<socket>.ready` marker are written beside the socket.
#
#   The boot uses THE TREE THIS SCRIPT LIVES IN and nothing else: its own
#   .build/release/bench-worker, its own weights/. MLXFAST_ENGINE_BIN and
#   MLXFAST_WEIGHTS_PATH are IGNORED here on purpose -- the ranked job exports
#   them for the candidate, and honouring them would boot the candidate's engine
#   for the reference leg, which is the bug this verb fixes.
#
# --stop:
#   --socket <path>    the socket a --boot reported. The stop reads
#                      <path>.pid, ends that resident, and removes the socket,
#                      the pid sidecar and the ready marker. It is IDEMPOTENT: a
#                      second stop, or a stop of a resident that already died,
#                      exits 0 and says so.
#
#   --weights <dir>    the transformed weights directory (`weights/` from
#                      setup.sh). Required.
#   --ngram <dir>      the n-gram row-source directory, passed to the resident
#                      as --resource qwen4exp.ngramRowSource=<dir>. Default: the
#                      weights directory (the fixture's ngram_shard_dir).
#   --hello-identity   export BENCH_WORKER_RESIDENT_HELLO=1 to the command, so
#                      each attached hello carries the resident's pid and
#                      load_epoch. Off by default until benchd admits the field.
#   --socket <path>    the resident's Unix socket. Default: a pid-keyed path
#                      under ${TMPDIR:-/tmp}. A Unix socket path holds 103
#                      bytes on macOS; a socket under the checkout would not
#                      bind on the ranked box.
#
# Environment (test seams and locations; none relaxes a refusal):
#   MLXFAST_ENGINE_BIN               the bench-worker binary benchd spawns
#                                    (default <repo>/.build/release/bench-worker,
#                                    the staged pair tools/stage-bench-worker.sh
#                                    writes, with mlx.metallib beside it; the
#                                    same variable and default the measure
#                                    script reads, so the resident and every
#                                    attaching phase run ONE binary)
#   RESIDENT_UP_LOG_DIR              pidfile, pgid file, identity file, resident
#                                    log (default <repo>/.build/resident)
#   RESIDENT_UP_LOCK_PATH            the GPU lock file (default
#                                    /tmp/mtplx-gpu-exclusive.lock)
#   RESIDENT_UP_HEALTH_TIMEOUT_S     ceiling on the load + first hello
#                                    (default 5400)
#   RESIDENT_UP_WIRED_LIMIT_READER   command printing iogpu.wired_limit_mb
#                                    (default `sysctl -n iogpu.wired_limit_mb`)
#
# Exit codes: the wrapped command's exit code on a served window; 1 on a boot
# failure; 2 on a refusal before any load.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
log() { printf 'resident-up.sh: %s\n' "$*" >&2; }
die() { printf 'resident-up.sh: %s\n' "$*" >&2; exit 1; }
refuse() { printf 'resident-up.sh: REFUSED (%s): %s\n' "$1" "$2" >&2; exit 2; }

usage() {
  cat >&2 <<'USAGE'
usage:
  tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N> --socket-out <file>
  tools/resident-up.sh --stop --socket <path>
  tools/resident-up.sh --weights <dir> [--ngram <dir>] [--hello-identity] [--socket <path>] -- <command> [args...]
USAGE
  exit 2
}

# --- argv --------------------------------------------------------------------
# MODE is the verb. `wrap` is the historical form and stays the default, so a
# hand run and the local unscored path are unchanged. --boot and --stop are the
# per-leg form benchd drives.
MODE="wrap"
WEIGHTS_DIR=""
NGRAM_DIR=""
SOCKET_PATH=""
SOCKET_OUT=""
SPEC=""
DRAFT_LEN=""
HELLO_IDENTITY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --boot) [[ "${MODE}" == "wrap" ]] || refuse bad-argument "--boot and --stop are exclusive"; MODE="boot"; shift ;;
    --stop) [[ "${MODE}" == "wrap" ]] || refuse bad-argument "--boot and --stop are exclusive"; MODE="stop"; shift ;;
    --spec) [[ $# -ge 2 ]] || refuse bad-argument "--spec needs a value (serial or mtp)"; SPEC="$2"; shift 2 ;;
    --draft-len) [[ $# -ge 2 ]] || refuse bad-argument "--draft-len needs a value"; DRAFT_LEN="$2"; shift 2 ;;
    --socket-out) [[ $# -ge 2 ]] || refuse bad-argument "--socket-out needs a path"; SOCKET_OUT="$2"; shift 2 ;;
    --weights) [[ $# -ge 2 ]] || refuse bad-argument "--weights needs a directory"; WEIGHTS_DIR="$2"; shift 2 ;;
    --ngram) [[ $# -ge 2 ]] || refuse bad-argument "--ngram needs a directory"; NGRAM_DIR="$2"; shift 2 ;;
    --socket) [[ $# -ge 2 ]] || refuse bad-argument "--socket needs a path"; SOCKET_PATH="$2"; shift 2 ;;
    --hello-identity) HELLO_IDENTITY=1; shift ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) refuse bad-argument "unknown argument '$1'" ;;
  esac
done

# --- --stop: idempotent teardown, and nothing else ---------------------------
# It runs before every other check on purpose. A teardown must work on a box
# whose GPU lock is gone, whose wired limit was never pinned, and whose resident
# already died -- the failure modes a stop exists to clean up after.
if [[ "${MODE}" == "stop" ]]; then
  [[ -n "${SOCKET_PATH}" ]] || refuse bad-argument "--stop needs --socket <path>"
  stop_pidfile="${SOCKET_PATH}.pid"
  stopped=""
  if [[ -r "${stop_pidfile}" ]]; then
    stop_pid="$(tr -d '[:space:]' < "${stop_pidfile}")"
    if [[ "${stop_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${stop_pid}" 2>/dev/null; then
      # The resident leads its own process group (it was setsid'd at boot), so
      # the group signal reaches anything it spawned.
      kill -TERM -- "-${stop_pid}" 2>/dev/null || kill -TERM "${stop_pid}" 2>/dev/null || true
      for _ in $(seq 1 50); do
        kill -0 "${stop_pid}" 2>/dev/null || break
        sleep 0.2
      done
      if kill -0 "${stop_pid}" 2>/dev/null; then
        log "resident ${stop_pid} did not stop on SIGTERM; sending SIGKILL to its group"
        kill -KILL -- "-${stop_pid}" 2>/dev/null || kill -KILL "${stop_pid}" 2>/dev/null || true
        sleep 0.3
      fi
      if kill -0 "${stop_pid}" 2>/dev/null; then
        die "FAILED to halt the resident: pid ${stop_pid} survived SIGKILL (socket ${SOCKET_PATH})"
      fi
      stopped="${stop_pid}"
    fi
  fi
  rm -f "${SOCKET_PATH}" "${stop_pidfile}" "${SOCKET_PATH}.ready" 2>/dev/null || true
  if [[ -n "${stopped}" ]]; then
    log "stopped the resident on ${SOCKET_PATH} (pid ${stopped}); socket, pid sidecar and ready marker removed"
  else
    log "nothing to stop on ${SOCKET_PATH} (no live resident); socket, pid sidecar and ready marker removed if present"
  fi
  exit 0
fi

# --- --boot: the leg's own tree, and only its own tree ------------------------
# THE FLAGS ARE AUTHORITATIVE. Nothing below reads mtp-head.manifest.json or
# tools/spec-declaration.sh: a reference tree whose manifest declares a depth
# still boots the SERIAL control leg when benchd says serial.
if [[ "${MODE}" == "boot" ]]; then
  [[ -n "${SPEC}" ]] || refuse bad-argument "--boot needs --spec <serial|mtp>"
  [[ -n "${DRAFT_LEN}" ]] || refuse bad-argument "--boot needs --draft-len <N>"
  [[ -n "${SOCKET_OUT}" ]] || refuse bad-argument "--boot needs --socket-out <file>"
  case "${SPEC}" in
    serial|mtp) : ;;
    *) refuse bad-argument "--spec must be 'serial' or 'mtp' (got '${SPEC}')" ;;
  esac
  [[ "${DRAFT_LEN}" =~ ^[0-9]+$ ]] || refuse bad-argument "--draft-len must be a non-negative integer (got '${DRAFT_LEN}')"
  if [[ "${SPEC}" == "serial" && "${DRAFT_LEN}" != "0" ]]; then
    refuse bad-argument "--spec serial requires --draft-len 0 (got ${DRAFT_LEN}); a serial control leg drafts nothing"
  fi
  if [[ "${SPEC}" == "mtp" ]] && { (( DRAFT_LEN < 1 )) || (( DRAFT_LEN > 6 )); }; then
    refuse bad-argument "--spec mtp requires --draft-len between 1 and 6 (got ${DRAFT_LEN}); the engine clamps at 6"
  fi
  # THE TREE THIS SCRIPT LIVES IN, and nothing the environment says. The ranked
  # job exports MLXFAST_ENGINE_BIN and MLXFAST_WEIGHTS_PATH for the CANDIDATE;
  # honouring either here would boot the candidate's engine or read the
  # candidate's weights for the reference leg.
  cd "${SCRIPT_DIR}"
  WEIGHTS_DIR="${SCRIPT_DIR}/weights"
  NGRAM_DIR="${SCRIPT_DIR}/weights"
fi

if [[ "${MODE}" == "wrap" ]]; then
  [[ $# -ge 1 ]] || refuse bad-argument "no command after '--'"
fi
[[ -n "${WEIGHTS_DIR}" ]] || refuse bad-argument "--weights <dir> is required"
[[ -d "${WEIGHTS_DIR}" ]] || refuse weights-missing "weights directory is missing: '${WEIGHTS_DIR}'"
NGRAM_DIR="${NGRAM_DIR:-${WEIGHTS_DIR}}"
[[ -d "${NGRAM_DIR}" ]] || refuse ngram-missing "n-gram row-source directory is missing: '${NGRAM_DIR}'"

command -v python3 >/dev/null 2>&1 || refuse tool-missing "python3 is required (it speaks the resident's hello probe and the lock check)"

# In --boot the binary is THIS TREE'S staged worker, full stop: an inherited
# MLXFAST_ENGINE_BIN names the candidate's, and the reference leg must not run
# it. The wrapper form keeps honouring the variable, because there the caller
# and the tree are the same thing.
if [[ "${MODE}" == "boot" ]]; then
  BENCH_WORKER="${SCRIPT_DIR}/.build/release/bench-worker"
  if [[ -n "${MLXFAST_ENGINE_BIN:-}" && "${MLXFAST_ENGINE_BIN}" != "${BENCH_WORKER}" ]]; then
    log "ignoring MLXFAST_ENGINE_BIN=${MLXFAST_ENGINE_BIN}; --boot runs THIS tree's worker (${BENCH_WORKER}), because the leg is defined by its tree"
  fi
else
  BENCH_WORKER="${MLXFAST_ENGINE_BIN:-${SCRIPT_DIR}/.build/release/bench-worker}"
fi
[[ -x "${BENCH_WORKER}" ]] || refuse worker-missing "bench-worker is missing or not executable: ${BENCH_WORKER} (build and stage it: setup.sh, or swift build -c release --scratch-path .build-worker --product track-bench-worker && MLXFAST_BENCH_WORKER_EXECUTABLE=.build-worker/release/track-bench-worker tools/stage-bench-worker.sh)"

LOG_DIR="${RESIDENT_UP_LOG_DIR:-${SCRIPT_DIR}/.build/resident}"
LOCK_PATH="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
HEALTH_TIMEOUT_S="${RESIDENT_UP_HEALTH_TIMEOUT_S:-5400}"
[[ "${HEALTH_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]] || refuse bad-argument "RESIDENT_UP_HEALTH_TIMEOUT_S must be a positive integer (got '${HEALTH_TIMEOUT_S}')"
WIRED_LIMIT_READER="${RESIDENT_UP_WIRED_LIMIT_READER:-sysctl -n iogpu.wired_limit_mb}"

mkdir -p "${LOG_DIR}"
RUN_TAG="$$-$(date -u +%Y%m%dT%H%M%SZ)"
SOCKET_PATH="${SOCKET_PATH:-${TMPDIR:-/tmp}/bench-worker-resident.${RUN_TAG}.sock}"
PIDFILE="${LOG_DIR}/resident.pid"
PGIDFILE="${LOG_DIR}/resident.pgid"
IDENTITY_FILE="${LOG_DIR}/resident-identity.json"
RESIDENT_LOG="${LOG_DIR}/resident.${RUN_TAG}.log"

# --- the GPU lock must be HELD by the caller ---------------------------------
# A non-blocking try-lock on a NEW open file description fails only when
# someone holds the lock. Success means nobody does: the caller is not inside
# a GPU window, and the try-lock is released at once.
lock_is_held() {
  python3 - "$1" <<'PY'
import fcntl, sys
try:
    f = open(sys.argv[1], "a+")
except OSError as err:
    print(f"cannot open the lock file: {err}", file=sys.stderr); sys.exit(2)
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(0)          # held by someone: the caller is inside a window
fcntl.flock(f, fcntl.LOCK_UN)
sys.exit(1)              # nobody holds it
PY
}
[[ -e "${LOCK_PATH}" ]] || refuse lock-missing "the GPU lock file does not exist: ${LOCK_PATH} (see the runbook, step 2)"
if ! lock_is_held "${LOCK_PATH}"; then
  refuse lock-not-held "nobody holds ${LOCK_PATH}; the caller must take the GPU lock for the whole window before booting a resident (flock, then run this script inside it). Nothing has been loaded."
fi

# --- the wired limit must be pinned ------------------------------------------
# The box runbook's boot daemon pins iogpu.wired_limit_mb so a 113 GB resident
# can be wired. This script verifies the pin and never sets it.
wired_limit_mb="$(${WIRED_LIMIT_READER} 2>/dev/null | tr -d '[:space:]' || true)"
if ! [[ "${wired_limit_mb}" =~ ^[0-9]+$ ]] || (( wired_limit_mb <= 0 )); then
  refuse wired-limit-unpinned "iogpu.wired_limit_mb is not pinned (reader '${WIRED_LIMIT_READER}' gave '${wired_limit_mb:-nothing}'); the boot daemon of the runbook sets it. Nothing has been loaded."
fi

# --- one resident per box ----------------------------------------------------
# The hello probe: connect, read the first line. The worker sends its hello
# unprompted at session start (hello is not a request kind in Engine Protocol
# v1), so a connection that receives an ok:true hello has found a live
# resident. The probe closes at once, which ends its session.
probe_resident() {
  python3 - "$1" <<'PY'
import json, socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(sys.argv[1])
    line = b""
    while not line.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        line += chunk
    s.close()
    reply = json.loads(line.decode())
except Exception as err:                       # noqa: BLE001 - any failure is unhealthy
    print(f"probe failed: {err}", file=sys.stderr)
    sys.exit(1)
if reply.get("ok") is not True:
    print(f"probe refused: {reply}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: reply.get(k) for k in
                  ("backend", "device", "protocol_version", "spec_modes", "runner", "resident")}))
PY
}

# ONE RESIDENT PER BOX, ACROSS TREES. The pidfile below lives under the tree
# that booted, so it cannot see a resident booted from the OTHER tree -- and
# under the paired design there are exactly two trees. A box-wide scan is what
# catches leg 2 booting before leg 1 was stopped, which would double-load 113 GB.
if [[ "${MODE}" == "boot" ]]; then
  live_residents="$(pgrep -f 'bench-worker resident' 2>/dev/null | tr '\n' ' ' | sed 's/ *$//' || true)"
  if [[ -n "${live_residents}" ]]; then
    refuse resident-already-up "a bench-worker resident is already running on this box (pid(s) ${live_residents}); the previous leg was not stopped. Stop it first (tools/resident-up.sh --stop --socket <path>). Nothing has been loaded."
  fi
fi

if [[ -r "${PIDFILE}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${PIDFILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    old_args="$(ps -o args= -p "${old_pid}" 2>/dev/null || true)"
    case "${old_args}" in
      *bench-worker*resident*)
        refuse resident-already-up "a resident is already running: pid ${old_pid} ('${old_args}'), recorded in ${PIDFILE}. One resident per box; halt it first (tools/resident-halt: SIGTERM the pgid in ${PGIDFILE}). Nothing has been loaded." ;;
      *)
        log "stale pidfile ${PIDFILE}: pid ${old_pid} is not a resident ('${old_args}'); removing it" ;;
    esac
  fi
  rm -f "${PIDFILE}" "${PGIDFILE}"
fi
if [[ -S "${SOCKET_PATH}" ]]; then
  if probe_resident "${SOCKET_PATH}" >/dev/null 2>&1; then
    refuse resident-already-up "a resident already answers on ${SOCKET_PATH}. One resident per box. Nothing has been loaded."
  fi
  log "stale socket ${SOCKET_PATH} (nothing answers); removing it"
  rm -f "${SOCKET_PATH}"
fi

# --- teardown ----------------------------------------------------------------
# ALWAYS. The resident holds the GPU and ~113 GB of unified memory, so it never
# outlives this script: the window that booted it tears it down, on success and
# on failure alike. SIGTERM to the process, then the process group, then
# SIGKILL to the group; the script then proves that nothing of the group is
# left. There is deliberately no path that leaves it up.
RESIDENT_PID=""
RESIDENT_PGID=""

group_members() { ps -o pid= -g "${RESIDENT_PGID}" 2>/dev/null | tr -d ' ' | grep -v '^$' || true; }

teardown() {
  if [[ -n "${RESIDENT_PID}" ]] && kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    kill -TERM "${RESIDENT_PID}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${RESIDENT_PID}" 2>/dev/null || break
      sleep 0.2
    done
  fi
  if [[ -n "${RESIDENT_PGID}" && "${RESIDENT_PGID}" != "$$" ]] && [[ -n "$(group_members)" ]]; then
    kill -TERM -- "-${RESIDENT_PGID}" 2>/dev/null || true
    for _ in $(seq 1 25); do
      [[ -n "$(group_members)" ]] || break
      sleep 0.2
    done
    if [[ -n "$(group_members)" ]]; then
      log "resident did not stop on SIGTERM; killing process group ${RESIDENT_PGID}"
      kill -KILL -- "-${RESIDENT_PGID}" 2>/dev/null || true
      sleep 0.3
    fi
    if [[ -n "$(group_members)" ]]; then
      log "FAILED to halt the resident: process(es) $(group_members | tr '\n' ' ')of group ${RESIDENT_PGID} survived SIGKILL"
    fi
  fi
  if [[ -n "${RESIDENT_PID}" ]]; then
    log "resident torn down; its log is ${RESIDENT_LOG}"
  fi
  rm -f "${SOCKET_PATH}" "${PIDFILE}" "${PGIDFILE}" 2>/dev/null || true
}
# THE TRAP IS THE WRAPPER FORM'S. A --boot deliberately EXITS with the resident
# still running -- that is the whole contract -- so it must not tear down on
# exit. Its failure paths call teardown explicitly instead, so a boot that never
# became healthy still leaves nothing behind.
if [[ "${MODE}" == "wrap" ]]; then
  trap teardown EXIT
fi
trap 'exit 130' INT TERM

# --- boot the one resident ---------------------------------------------------
# The resident leads its own process group (setsid through python, which macOS
# has and util-linux setsid it does not), so the halt path can signal the
# group and reach anything the worker spawned.
if [[ "${MODE}" == "boot" ]]; then
  log "booting the ${SPEC} leg's resident from ${SCRIPT_DIR}: ${BENCH_WORKER} resident --weights ${WEIGHTS_DIR} (n-gram rows from ${NGRAM_DIR}, declared draft length ${DRAFT_LEN}); ONE load for THIS leg"
else
  log "booting the resident: ${BENCH_WORKER} resident --weights ${WEIGHTS_DIR} (n-gram rows from ${NGRAM_DIR}); ONE load for the whole window"
fi
python3 - "${BENCH_WORKER}" resident \
  --weights "${WEIGHTS_DIR}" \
  --speculative-protocol v1.1 \
  --resource "qwen4exp.ngramRowSource=${NGRAM_DIR}" \
  --socket "${SOCKET_PATH}" >"${RESIDENT_LOG}" 2>&1 <<'PY' &
import os, sys
os.setsid()
os.execv(sys.argv[1], sys.argv[1:])
PY
RESIDENT_PID=$!
RESIDENT_PGID="${RESIDENT_PID}"
printf '%s\n' "${RESIDENT_PID}" > "${PIDFILE}"
printf '%s\n' "${RESIDENT_PGID}" > "${PGIDFILE}"

start="$(date +%s)"
while :; do
  if ! kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    log "the resident exited before it was healthy; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    [[ "${MODE}" != "boot" ]] || teardown
    exit 1
  fi
  if [[ -S "${SOCKET_PATH}" ]] && probe_resident "${SOCKET_PATH}" >/dev/null 2>&1; then
    break
  fi
  now="$(date +%s)"
  if (( now - start >= HEALTH_TIMEOUT_S )); then
    log "the resident was not healthy within ${HEALTH_TIMEOUT_S}s; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    [[ "${MODE}" != "boot" ]] || teardown
    exit 1
  fi
  sleep 2
done
log "resident healthy on ${SOCKET_PATH} after $(( $(date +%s) - start ))s; every phase now attaches instead of loading"

HELLO_JSON="$(probe_resident "${SOCKET_PATH}")" || die "the resident stopped answering before the run started"

# --- the window's identity ---------------------------------------------------
# Every value crosses as an argv element, never as text spliced into the Python
# source: a socket path or a weights directory holding a quote would otherwise
# be shell string interpolation inside a program, and the file this run is
# identified by would stop parsing (or stop meaning what it says).
python3 - "${IDENTITY_FILE}" "${HELLO_JSON}" \
    "${SOCKET_PATH}" "${RESIDENT_PID}" "${RESIDENT_PGID}" \
    "${BENCH_WORKER}" "${WEIGHTS_DIR}" "${NGRAM_DIR}" \
    "${HELLO_IDENTITY}" "${MODE}" "${SPEC:-unset}" "${DRAFT_LEN:-unset}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" <<'PY'
import json, sys
(identity_file, hello_json, socket_path, resident_pid, resident_pgid,
 bench_worker, weights_dir, ngram_dir, hello_identity, mode, spec, draft_len,
 started) = sys.argv[1:14]
json.dump({
    "engine": "bench-worker",
    "weight_owner": "bench-worker-resident",
    "resident_socket": socket_path,
    "resident_pid": int(resident_pid),
    "resident_pgid": int(resident_pgid),
    "bench_worker": bench_worker,
    "weights_dir": weights_dir,
    "ngram_dir": ngram_dir,
    "hello_identity": int(hello_identity),
    "mode": mode,
    "spec": spec,
    "draft_len": draft_len,
    "hello": json.loads(hello_json),
    "started": started,
}, open(identity_file, "w"), indent=2)
PY

# --- --boot ends here, with the resident STILL RUNNING -----------------------
# The contract with benchd: the socket path is the FIRST LINE of --socket-out,
# a <socket>.pid sidecar names the process --stop must end, and a
# <socket>.ready marker records that the hello was healthy. Then exit 0 and
# leave it up. The resident was setsid'd at boot, so it does not die with this
# script.
if [[ "${MODE}" == "boot" ]]; then
  mkdir -p "$(dirname "${SOCKET_OUT}")"
  printf '%s\n' "${SOCKET_PATH}" > "${SOCKET_OUT}"
  printf '%s\n' "${RESIDENT_PID}" > "${SOCKET_PATH}.pid"
  printf '%s %s %s\n' "${SPEC}" "${DRAFT_LEN}" "${RESIDENT_PID}" > "${SOCKET_PATH}.ready"
  log "boot complete: ${SPEC} leg resident pid ${RESIDENT_PID} on ${SOCKET_PATH}; socket written to ${SOCKET_OUT}. Stop it with: tools/resident-up.sh --stop --socket ${SOCKET_PATH}"
  exit 0
fi

export BENCH_WORKER_RESIDENT_SOCKET="${SOCKET_PATH}"
if [[ "${HELLO_IDENTITY}" == "1" ]]; then
  export BENCH_WORKER_RESIDENT_HELLO=1
else
  unset BENCH_WORKER_RESIDENT_HELLO || true
fi
export RESIDENT_IDENTITY_FILE="${IDENTITY_FILE}"
export RESIDENT_UP_PID="${RESIDENT_PID}"

log "running: $*"
set +e
"$@"
rc=$?
set -e
exit "${rc}"
