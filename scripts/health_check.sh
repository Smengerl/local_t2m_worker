#!/usr/bin/env bash
set -euo pipefail

# Cyclic health check — refreshes every ~5 s until Ctrl-C.
# Prints six status lines per cycle:
#  * Batch worker running   (PID file + kill -0)
#  * Server process running (pgrep batch.server)
#  * Server reachable       (HTTP probe on localhost)
#  * Generation running     (job with status=running in queue.jsonl)
#  * HF download rate       (RX on external interface, e.g. en0/wifi)
#  * Web GUI traffic rate   (RX on loopback lo0 = localhost-only traffic)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE_FILE="$ROOT_DIR/queue.jsonl"
PORT="${PORT:-8000}"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"

# ── helpers ───────────────────────────────────────────────────────────────────

status() {
  if [[ $1 == true ]]; then
    printf "  ✔ %s\n" "$2"
  else
    printf "  ✖ %s\n" "$2"
  fi
}

get_rx() {
  # macOS: netstat -ib, column 7 (Ibytes) on the <Link#> row for a given interface
  netstat -ib 2>/dev/null | awk -v iface="$1" '$1==iface && /Link/ {print $7; exit}' || echo 0
}

fmt_rate() {
  # Format bytes/s as human-readable string
  local bytes=$1
  if   (( bytes >= 1048576 )); then printf "%.1f MB/s" "$(echo "scale=1; $bytes/1048576" | bc)"
  elif (( bytes >= 1024 ));    then printf "%d KiB/s"  "$(( bytes / 1024 ))"
  else                              printf "%d B/s"    "$bytes"
  fi
}

# Detect primary external network interface once at startup
ext_iface=$(route get default 2>/dev/null | awk '/interface:/ {print $2; exit}')
ext_iface="${ext_iface:-en0}"

run_checks() {
  # 1) Worker alive? + pipeline/queue details — all from GET /api/health
  worker_ok=false
  worker_error=""
  current_job=""
  pipeline_cached=false
  loaded_model=""
  q_pending=0; q_running=0; q_done=0; q_failed=0

  health_json=$(curl -fsS --max-time 2 "http://localhost:$PORT/api/health" 2>/dev/null || true)
  if [[ -n "$health_json" ]]; then
    _jq() { echo "$health_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d$1)" 2>/dev/null || true; }
    [[ "$(_jq "['worker_alive']")" == "True" ]] && worker_ok=true
    worker_error=$(_jq "['worker_error'] or ''")
    current_job=$(_jq "['current_job_id'] or ''")
    [[ "$(_jq "['pipeline_cached']")" == "True" ]] && pipeline_cached=true
    loaded_model=$(_jq "['loaded_model'] or ''")
    q_pending=$(_jq "['queue']['pending']")
    q_running=$(_jq "['queue']['running']")
    q_done=$(_jq "['queue']['done']")
    q_failed=$(_jq "['queue']['failed']")
  fi

  # Derive generation_ok from the queue data we already have
  generation_ok=false
  [[ "${q_running:-0}" -gt 0 ]] && generation_ok=true

  # 2) Server process running? (pgrep matches -m batch.server in argv)
  server_proc_ok=false
  if pgrep -f "batch\.server" >/dev/null 2>&1; then
    server_proc_ok=true
  fi

  # 3) Server reachable? (HTTP probe on localhost)
  server_http_ok=false
  if curl -fsS --max-time 2 "http://localhost:$PORT/" >/dev/null 2>&1; then
    server_http_ok=true
  fi

  # 4) Network rates — sample RX on external iface (HF) and lo0 (web GUI) over 1 s
  ext_rx1=$(get_rx "$ext_iface"); lo_rx1=$(get_rx "lo0")
  sleep 1
  ext_rx2=$(get_rx "$ext_iface"); lo_rx2=$(get_rx "lo0")

  ext_rate=0; lo_rate=0
  [[ "$ext_rx1" =~ ^[0-9]+$ && "$ext_rx2" =~ ^[0-9]+$ ]] && ext_rate=$(( ext_rx2 - ext_rx1 ))
  [[ "$lo_rx1"  =~ ^[0-9]+$ && "$lo_rx2"  =~ ^[0-9]+$ ]] && lo_rate=$(( lo_rx2  - lo_rx1  ))

  hf_ok=false;  [[ $ext_rate -gt 1024 ]] && hf_ok=true
  gui_ok=false; [[ $lo_rate  -gt 1024 ]] && gui_ok=true

  # ── output ────────────────────────────────────────────────────────────────
  clear
  printf "  Local T2M Worker  —  %s  (Ctrl-C to quit)\n\n" "$(date '+%H:%M:%S')"

  status $worker_ok      "Worker task alive"
  if ! $worker_ok && [[ -n "$worker_error" && "$worker_error" != "None" ]]; then
    printf "    ↳ %s\n" "$worker_error"
  fi
  status $server_proc_ok "Server process running"
  status $server_http_ok "Server reachable (localhost:$PORT)"
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

  printf "\n  Network\n"
  if $hf_ok; then
    printf "  ✔ HF download   $(fmt_rate $ext_rate) (via %s)\n" "$ext_iface"
  else
    printf "  ✖ HF download   $(fmt_rate $ext_rate) (via %s)\n" "$ext_iface"
  fi
  if $gui_ok; then
    printf "  ✔ Web GUI traffic  $(fmt_rate $lo_rate) (lo0)\n"
  else
    printf "  ✖ Web GUI traffic  $(fmt_rate $lo_rate) (lo0)\n"
  fi
}

# ── main loop ─────────────────────────────────────────────────────────────────

trap 'printf "\nStopped.\n"; exit 0' INT TERM

while true; do
  run_checks
  # refresh every 5 s
  sleep 5
done
