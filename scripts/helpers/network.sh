#!/usr/bin/env bash
# helpers/network.sh — Cross-platform network utility functions
#
# Functions provided:
#   detect_ext_iface      — sets EXT_IFACE to the primary external network interface
#   get_rx IFACE          — prints current RX bytes for an interface
#   get_loopback_iface    — prints the loopback interface name (lo0 on macOS, lo on Linux)
#   fmt_rate BYTES        — formats bytes/s as human-readable string
#
# USAGE: Source this file after platform.sh is loaded.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT_DIR/scripts/helpers/platform.sh"

# ── detect_ext_iface ──────────────────────────────────────────────────────────
# Sets EXT_IFACE to the primary outbound network interface name.
detect_ext_iface() {
  if is_macos; then
    EXT_IFACE=$(route get default 2>/dev/null | awk '/interface:/ {print $2; exit}')
  elif is_linux; then
    EXT_IFACE=$(ip route show default 2>/dev/null | awk '/default/ {print $5; exit}')
  else
    # Windows / fallback
    EXT_IFACE="Ethernet"
  fi
  EXT_IFACE="${EXT_IFACE:-en0}"
}

# ── get_loopback_iface ────────────────────────────────────────────────────────
get_loopback_iface() {
  if is_macos; then
    echo "lo0"
  else
    echo "lo"
  fi
}

# ── get_rx IFACE ──────────────────────────────────────────────────────────────
# Prints current cumulative RX bytes for a network interface.
get_rx() {
  local iface="$1"
  if is_macos; then
    netstat -ib 2>/dev/null \
      | awk -v iface="$iface" '$1==iface && /Link/ {print $7; exit}' || echo 0
  elif is_linux; then
    cat "/sys/class/net/$iface/statistics/rx_bytes" 2>/dev/null || echo 0
  else
    # Windows: not supported via bash — return 0
    echo 0
  fi
}

# ── fmt_rate BYTES ────────────────────────────────────────────────────────────
# Formats a bytes-per-second value as a human-readable string.
fmt_rate() {
  local bytes=$1
  if   (( bytes >= 1048576 )); then printf "%.1f MB/s" "$(echo "scale=1; $bytes/1048576" | bc)"
  elif (( bytes >= 1024 ));    then printf "%d KiB/s"  "$(( bytes / 1024 ))"
  else                              printf "%d B/s"    "$bytes"
  fi
}
