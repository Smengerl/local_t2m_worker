#!/usr/bin/env bash
# health_check.sh — Cyclic health display. Can also be sourced to get run_checks().
#
# Standalone: ./scripts/health_check.sh   — loops until Ctrl-C
# Sourceable: source .../health_check.sh  — provides run_checks(), requires
#             ROOT_DIR, PYTHON, SYS_PYTHON, EXT_IFACE, LOOPBACK_IFACE to be set

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"
# shellcheck source=helpers/worker_status.sh
source "$ROOT_DIR/scripts/helpers/worker_status.sh"
# shellcheck source=helpers/network.sh
source "$ROOT_DIR/scripts/helpers/network.sh"

PORT="${PORT:-8000}"
WORKER_LOG="${WORKER_LOG:-$ROOT_DIR/batch/worker.log}"

# Activate venv and resolve PYTHON so batch_instance_running() can spawn Python.
activate_venv
resolve_venv_python

# Detect interfaces (safe to call multiple times — no-op if already set)
detect_ext_iface
LOOPBACK_IFACE=$(get_loopback_iface)

# ── helpers ───────────────────────────────────────────────────────────────────

status() {
  if [[ $1 == true ]]; then
    printf "  ✔ %s\n" "$2"
  else
    printf "  ✖ %s\n" "$2"
  fi
}

run_checks() {

  # 0+1) batch.server OR batch.worker running? (single lock-based check)
  batch_proc_ok=false
  if batch_instance_running; then
    batch_proc_ok=true
  fi

  # 2) Server reachable? (HTTP probe on localhost)
  server_http_ok=false
  if curl -fsS --max-time 2 "http://localhost:$PORT/" >/dev/null 2>&1; then
    server_http_ok=true
  fi

  # 3) Only if reachable: query health endpoint for worker/queue/model info
  worker_ok=false
  worker_error=""
  current_job=""
  pipeline_cached=false
  loaded_model=""
  q_pending=0; q_running=0; q_done=0; q_failed=0
  generation_ok=false
  if $server_http_ok; then
    health_json=$(curl -fsS --max-time 2 "http://localhost:$PORT/api/health" 2>/dev/null || true)
    if [[ -n "$health_json" ]]; then
      _jq() { echo "$health_json" | "$SYS_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d$1)" 2>/dev/null || true; }
      [[ "$(_jq "['worker_alive']")" == "True" ]] && worker_ok=true
      worker_error=$(_jq "['worker_error'] or ''")
      current_job=$(_jq "['current_job_id'] or ''")
      [[ "$(_jq "['pipeline_cached']")" == "True" ]] && pipeline_cached=true
      loaded_model=$(_jq "['loaded_model'] or ''")
      q_pending=$(_jq "['queue']['pending']")
      q_running=$(_jq "['queue']['running']")
      q_done=$(_jq "['queue']['done']")
      q_failed=$(_jq "['queue']['failed']")
      [[ "${q_running:-0}" -gt 0 ]] && generation_ok=true
    fi
  fi

  # 4) Network rates — sample RX over 1 s
  ext_rx1=$(get_rx "$EXT_IFACE"); lo_rx1=$(get_rx "$LOOPBACK_IFACE")
  sleep 1
  ext_rx2=$(get_rx "$EXT_IFACE"); lo_rx2=$(get_rx "$LOOPBACK_IFACE")

  ext_rate=0; lo_rate=0
  [[ "$ext_rx1" =~ ^[0-9]+$ && "$ext_rx2" =~ ^[0-9]+$ ]] && ext_rate=$(( ext_rx2 - ext_rx1 ))
  [[ "$lo_rx1"  =~ ^[0-9]+$ && "$lo_rx2"  =~ ^[0-9]+$ ]] && lo_rate=$(( lo_rx2  - lo_rx1  ))

  hf_ok=false;  [[ $ext_rate -gt 1024 ]] && hf_ok=true
  gui_ok=false; [[ $lo_rate  -gt 1024 ]] && gui_ok=true

  # ── output ────────────────────────────────────────────────────────────────

  clear
  printf "  Local T2M Worker Server Health Check —  %s  (Ctrl-C to quit)\n\n" "$(date '+%H:%M:%S')"

  status $batch_proc_ok  "batch.worker / batch.server running (batch.lock)"
  status $server_http_ok "Server reachable (localhost:$PORT)"
  if $server_http_ok; then
    status $worker_ok      "Server batch worker task alive"
    if ! $worker_ok && [[ -n "$worker_error" && "$worker_error" != "None" ]]; then
      printf "    ↳ %s\n" "$worker_error"
    fi
    status $generation_ok  "Generation running"
    if $generation_ok && [[ -n "$current_job" && "$current_job" != "None" ]]; then
      printf "    ↳ job %s\n" "${current_job:0:8}"
    fi

    printf "\n  Pipeline cache\n"
    if $pipeline_cached && [[ -n "$loaded_model" && "$loaded_model" != "None" ]]; then
      printf "  ✔ Loaded   %s\n" "$loaded_model"
    else
      printf "  ✖ Empty    (no model in memory)\n"
    fi

    printf "\n  Queue\n"
    printf "    pending  %s\n" "${q_pending:-0}"
    printf "    running  %s\n" "${q_running:-0}"
    printf "    done     %s\n" "${q_done:-0}"
    printf "    failed   %s\n" "${q_failed:-0}"
  fi

  printf "\n  Network\n"
  status $hf_ok  "HF download   $(fmt_rate $ext_rate) (via $EXT_IFACE)"
  status $gui_ok "Web GUI traffic  $(fmt_rate $lo_rate) ($LOOPBACK_IFACE)"

  # ── worker log tail ───────────────────────────────────────────────────────
  if $batch_proc_ok && [[ -f "$WORKER_LOG" ]]; then
    printf "\n  Worker log  (last 8 lines — %s)\n" "$WORKER_LOG"
    printf "  %s\n" "$(printf '─%.0s' {1..60})"
    tail -8 "$WORKER_LOG" | while IFS= read -r line; do
      printf "  %s\n" "$line"
    done
  fi
}

# ── main loop ─────────────────────────────────────────────────────────────────
# Only run the loop when executed directly — not when sourced by _helper.sh.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  trap 'printf "\nStopped.\n"; exit 0' INT TERM
  while true; do
    run_checks
    sleep 5
  done
fi
