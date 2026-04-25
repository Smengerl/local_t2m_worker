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
PID_FILE="$ROOT_DIR/batch/worker.pid"
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
  # 1) Batch worker running? (PID file + process liveness)
  worker_ok=false
  if [[ -f "$PID_FILE" ]]; then
    PID=$(<"$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      worker_ok=true
    fi
  fi

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

  # 4) Generation running? (job.status == "running" in queue.jsonl)
  generation_ok=false
  if [[ -f "$QUEUE_FILE" ]]; then
    if command -v jq >/dev/null 2>&1; then
      # --slurp (-s) reads all JSONL lines into a single array before filtering
      running_count=$(jq -s '[.[] | select(.status=="running")] | length' "$QUEUE_FILE" 2>/dev/null || echo 0)
      [[ "$running_count" -gt 0 ]] && generation_ok=true
    else
      running=$("$SYS_PYTHON" - "$QUEUE_FILE" <<'PY'
import json, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    j = json.loads(line)
    if j.get('status') == 'running':
        print(1)
        break
else:
    print(0)
PY
)
      [[ "$running" == "1" ]] && generation_ok=true
    fi
  fi

  # 5) Network rates — sample RX on external iface (HF) and lo0 (web GUI) over 1 s
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
  status $worker_ok      "Batch worker running"
  status $server_proc_ok "Server process running"
  status $server_http_ok "Server reachable (localhost:$PORT)"
  status $generation_ok  "Generation running"
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
