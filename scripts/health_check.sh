#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"
# shellcheck source=helpers/worker_status.sh
source "$ROOT_DIR/scripts/helpers/worker_status.sh"
# shellcheck source=helpers/network.sh
source "$ROOT_DIR/scripts/helpers/network.sh"

# Cyclic health check — refreshes every ~5 s until Ctrl-C.
# Prints six status lines per cycle:
#  * Batch worker running   (PID file + kill -0)
#  * Server process running (pgrep / wmic batch.server)
#  * Server reachable       (HTTP probe on localhost)
#  * Generation running     (job with status=running in queue.jsonl)
#  * HF download rate       (RX on external interface)
#  * Web GUI traffic rate   (RX on loopback)

QUEUE_FILE="$ROOT_DIR/queue.jsonl"
PORT="${PORT:-8000}"

# Detect interfaces once at startup
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

  # 0) Local CLI worker running?
  local_worker_ok=false
  if worker_running; then
    local_worker_ok=true
  fi

  # 1) Server process running?
  server_proc_ok=false
  if server_process_running; then
    server_proc_ok=true
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

  status $local_worker_ok "Local CLI worker running (batch.worker)"
  status $server_proc_ok "Server process running"
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
}

# ── main loop ─────────────────────────────────────────────────────────────────

trap 'printf "\nStopped.\n"; exit 0' INT TERM

while true; do
  run_checks
  sleep 5
done
  run_checks
  # refresh every 5 s
  sleep 5
done
