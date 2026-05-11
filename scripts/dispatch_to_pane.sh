#!/usr/bin/env bash
# scripts/dispatch_to_pane.sh — canonical dispatch primitive for Codex tmux panes.
#
# Origin: 2026-05-11 T-ROOT-COORD Track O. Promotes empirically-validated
# chunk=1000 + double-Enter + immediate-recovery candidate from
# scripts/dispatch_to_pane.sh.candidate_a (kept as historical archive).
#
# Track AD (2026-05-11): adds opt-in pre-warm mode per Track AC v2.1 design to
# bypass fresh-Codex-session startup latency (observed 15-290 s across today's
# session rollouts; cf. eval_runs/codex_panes_evidence_2026-05-10/track_ac_prewarm_design_2026-05-11.md).
#
# Empirical basis (visible_ack-only metric, Opt-c 3-guard):
#   - iter-2 (10 trials, fresh pane mixed candidates): chunk=4000 first-attempt 0/6;
#     chunk=1000 first-attempt 1/1 at 4 KB; chunk=200 first-attempt 2/2 at 4 KB.
#   - iter-3 (2 trials, fresh pane): chunk=1000 first-attempt 2/2 at 8 KB / 12 KB.
#   - iter-4 (9 trials, single long-lived pane mixed 4/8/12 KB): chunk=1000
#     first-attempt 8/9 (89%); single failure was the very first dispatch in a
#     fresh codex pane (MCP startup-timeout suspect).
#   - Cumulative chunk=1000: 11/12 = 91.7% first-attempt success across 4-12 KB.
#
# Recovery primitives (immediate, embedded):
#   - Draft state (› prefix on payload, no Working): double-Enter recovery.
#   - Plan-mode dialog ("Create a plan? shift + tab use Plan mode esc dismiss"):
#     Esc + double-Enter recovery.
#   - Up to DISPATCH_TO_PANE_MAX_RECOVERY (default 3) attempts per phase.
#
# Usage:
#   dispatch_to_pane PANE_ID PAYLOAD [ACK_MARKER] [TIMEOUT_SEC] [--prewarm | --no-prewarm] [--warmup-timeout N]
#   dispatch_to_pane PANE_ID - [ACK_MARKER] [TIMEOUT_SEC] [flags]   # PAYLOAD from stdin
#
# Flags (Track AD v2.1):
#   --prewarm           — enable pre-warm phase for this invocation
#   --no-prewarm        — explicitly disable pre-warm (overrides DISPATCH_TO_PANE_PREWARM env)
#   --warmup-timeout N  — override warmup-phase ACK wait (non-negative integer; default 360)
#
# Pre-warm precedence (highest → lowest):
#   1. --no-prewarm flag             (explicit per-invocation OFF)
#   2. --prewarm flag                (explicit per-invocation ON)
#   3. DISPATCH_TO_PANE_PREWARM=0|1  (env)
#   4. default                       (DISABLED)
#
# `--` end-of-options convention (IMPORTANT for caller scripts):
#   Tokens after `--` are positional regardless of content (including those that
#   begin with `-` or equal a known flag token). Use `--` when PAYLOAD or
#   ACK_MARKER content might be mistaken for a flag. Examples:
#     dispatch_to_pane %68 -- "--prewarm" "TEST_ACK_X"     # literal "--prewarm" payload
#     dispatch_to_pane %68 --prewarm "abc" "TEST_ACK_X"    # enable pre-warm + send "abc"
#   Dynamically-generated payload/marker: always insert `--` before the positional
#   payload to disambiguate.
#
# Required args:
#   PANE_ID   — tmux pane id (e.g. %68 or session:window.pane).
#   PAYLOAD   — literal payload string, or "-" to read from stdin.
#
# Optional positional:
#   ACK_MARKER  — exact line text expected after Codex `^•` prefix; if omitted,
#                 returns after submission verification only (no ACK wait).
#   TIMEOUT_SEC — total main-ACK wait deadline in seconds; default 240.
#                 Note: with --prewarm the main timeout starts AFTER warmup ACK,
#                 so a warm pane typically completes in 60-90 s. Without --prewarm
#                 a fresh Codex pane's first dispatch can exceed 180 s before
#                 Codex registers the user turn (Track AA failure mode).
#
# Environment overrides:
#   DISPATCH_TO_PANE_CHUNK             — chunk size for literal send (default 1000)
#   DISPATCH_TO_PANE_SLEEP_S           — inter-Enter sleep (default 1)
#   DISPATCH_TO_PANE_VERIFY_S          — initial post-dispatch sleep (default 5)
#   DISPATCH_TO_PANE_MAX_RECOVERY      — max recovery attempts per phase (default 3)
#   DISPATCH_TO_PANE_PREWARM           — 0 (off, default) or 1 (on); overridden by flags
#   DISPATCH_TO_PANE_WARMUP_TIMEOUT_S  — warmup-phase ACK wait (default 360)
#
# Exit codes:
#   0  — visible ACK received within TIMEOUT_SEC (or no ACK_MARKER, submission OK)
#   1  — submission OK (Working observed) but main ACK timeout
#   2  — invalid arguments (e.g. --prewarm + --no-prewarm conflict, bad --warmup-timeout)
#   3  — recovery exhausted in main phase, no submission detected
#   4  — tmux command failure; phase= field indicates warmup or main
#   5  — pre-warm logical failure (warmup ACK / recovery did not complete though
#        tmux worked); main payload was NEVER dispatched
#
# Stderr emits one structured line on completion:
#   DISPATCH_TO_PANE_RESULT pane=<id> bytes=<n> phase=<warmup|main> warmup_seconds=<n|0> first_attempt=<YES|YES_WORKING|NO|ERROR> recovery=<none|double_enter|esc_plan_dismiss|...> rcount=<n> plan_dialog=<true|false> ack=<YES|NO|UNKNOWN> seconds=<n>
#
# `bytes=` semantics: actual byte count of the captured main PAYLOAD (not bash
# character count — UTF-8 multibyte characters are counted as multiple bytes),
# regardless of phase. When phase=warmup, the warmup-phase tiny payload
# (~120 bytes) is NOT reflected in this field; `bytes=` lets callers know how
# big the user-supplied content was, independent of which phase failed.
#
# Note: when PAYLOAD is read from stdin ("-"), bash command substitution
# (`PAYLOAD="$(cat)"`) strips trailing newlines before this count is taken,
# so a file ending with a single trailing newline will report (file_size - 1).

set -uo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  dispatch_to_pane PANE_ID PAYLOAD [ACK_MARKER] [TIMEOUT_SEC] [--prewarm | --no-prewarm] [--warmup-timeout N]
  dispatch_to_pane PANE_ID - [ACK_MARKER] [TIMEOUT_SEC] [flags]

Flags:
  --prewarm           enable pre-warm phase (Codex startup warmup turn before main payload)
  --no-prewarm        explicitly disable pre-warm (overrides DISPATCH_TO_PANE_PREWARM env)
  --warmup-timeout N  override warmup-phase ACK wait (non-negative integer; default 360)

Use `--` to mark end-of-options if PAYLOAD or ACK_MARKER may start with `-`
or equal a known flag token, e.g.:
  dispatch_to_pane %68 -- "--prewarm" "TEST_ACK_X"     # payload literal is "--prewarm"
  dispatch_to_pane %68 --prewarm "abc" "TEST_ACK_X"    # enable pre-warm + send "abc"
USAGE
  exit 2
}

# === argument parser: collect flags + positionals ===
# Walk tokens; recognize known flag tokens; treat anything after `--` as positional.
declare -a POSITIONAL=()
PREWARM_FLAG=""        # "" unset, "1" enable, "0" disable
WARMUP_TIMEOUT_FLAG="" # "" unset (env / default applies)

end_of_opts="false"
while (( $# > 0 )); do
  if [[ "$end_of_opts" == "true" ]]; then
    POSITIONAL+=("$1")
    shift
    continue
  fi
  case "$1" in
    --)
      end_of_opts="true"
      shift
      ;;
    --prewarm)
      [[ "$PREWARM_FLAG" == "0" ]] && { printf 'dispatch_to_pane: --prewarm conflicts with prior --no-prewarm\n' >&2; exit 2; }
      PREWARM_FLAG="1"
      shift
      ;;
    --no-prewarm)
      [[ "$PREWARM_FLAG" == "1" ]] && { printf 'dispatch_to_pane: --no-prewarm conflicts with prior --prewarm\n' >&2; exit 2; }
      PREWARM_FLAG="0"
      shift
      ;;
    --warmup-timeout)
      shift
      [[ $# -lt 1 ]] && { printf 'dispatch_to_pane: --warmup-timeout requires an integer argument\n' >&2; exit 2; }
      if ! [[ "$1" =~ ^[0-9]+$ ]]; then
        printf 'dispatch_to_pane: --warmup-timeout must be a non-negative integer (got: %s)\n' "$1" >&2
        exit 2
      fi
      WARMUP_TIMEOUT_FLAG="$1"
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

(( ${#POSITIONAL[@]} >= 2 )) || usage

PANE_ID="${POSITIONAL[0]}"
PAYLOAD_ARG="${POSITIONAL[1]}"
ACK_MARKER="${POSITIONAL[2]:-}"
TIMEOUT_SEC="${POSITIONAL[3]:-240}"

# === env / default reads ===
CHUNK_SIZE="${DISPATCH_TO_PANE_CHUNK:-1000}"
SLEEP_BETWEEN_ENTERS="${DISPATCH_TO_PANE_SLEEP_S:-1}"
INITIAL_VERIFY_SLEEP="${DISPATCH_TO_PANE_VERIFY_S:-5}"
MAX_RECOVERY="${DISPATCH_TO_PANE_MAX_RECOVERY:-3}"

# Validate non-negative integer; clear error + exit 2 if malformed.
# Used for env-supplied and positional integer fields so that an invalid value
# is caught BEFORE any bash arithmetic context (which under `set -u` raises
# "unbound variable" and bypasses the designed exit-2 path).
validate_non_negative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    printf 'dispatch_to_pane: %s must be a non-negative integer (got: %s)\n' "$name" "$value" >&2
    exit 2
  fi
}

# Validate TIMEOUT_SEC (positional or default) BEFORE arithmetic.
validate_non_negative_int "TIMEOUT_SEC (positional / default)" "$TIMEOUT_SEC"

# Pre-warm precedence ladder
if [[ "$PREWARM_FLAG" == "0" || "$PREWARM_FLAG" == "1" ]]; then
  PREWARM_ENABLED="$PREWARM_FLAG"
else
  case "${DISPATCH_TO_PANE_PREWARM:-0}" in
    0|1) PREWARM_ENABLED="${DISPATCH_TO_PANE_PREWARM:-0}" ;;
    *)   PREWARM_ENABLED="0" ;;  # malformed env -> safe default
  esac
fi

if [[ -n "$WARMUP_TIMEOUT_FLAG" ]]; then
  WARMUP_TIMEOUT_SEC="$WARMUP_TIMEOUT_FLAG"
else
  WARMUP_TIMEOUT_SEC="${DISPATCH_TO_PANE_WARMUP_TIMEOUT_S:-360}"
fi

# Validate WARMUP_TIMEOUT_SEC (flag overrides already validated by parser;
# this catches the env-supplied path that bypasses the parser).
validate_non_negative_int "DISPATCH_TO_PANE_WARMUP_TIMEOUT_S / --warmup-timeout" "$WARMUP_TIMEOUT_SEC"

[[ -z "$PANE_ID" ]] && { printf 'dispatch_to_pane: missing tmux pane id\n' >&2; exit 2; }

if [[ "$PAYLOAD_ARG" == "-" ]]; then
  PAYLOAD="$(cat)"
else
  PAYLOAD="$PAYLOAD_ARG"
fi

# LEN: actual byte count of the captured main PAYLOAD. Drives the bytes=
# field in the structured RESULT line. Computed after stdin capture (which
# strips trailing newlines via bash command substitution); LC_ALL=C and
# `wc -c` guarantee byte-count semantics regardless of locale.
LEN=$(LC_ALL=C printf '%s' "$PAYLOAD" | wc -c | tr -d ' ')
# CHAR_LEN: bash character count, used for char-indexed chunking via
# parameter substring expansion (${PAYLOAD:OFF:CHUNK_SIZE}). For ASCII-only
# payloads CHAR_LEN == LEN; UTF-8 multibyte payloads have CHAR_LEN < LEN.
CHAR_LEN=${#PAYLOAD}
T0=$(date +%s)
WARMUP_T0=0
WARMUP_ELAPSED=0
CURRENT_PHASE="main"   # set to "warmup" during pre-warm phase

# === shared helpers ===
capture_state() {
  tmux capture-pane -p -t "$PANE_ID" -S - 2>/dev/null | tail -n 30
}

ack_present_in_full() {
  local marker="$1"
  [[ -z "$marker" ]] && return 1
  tmux capture-pane -p -t "$PANE_ID" -S - 2>/dev/null \
    | tail -n 250 \
    | grep -Eq "^•[[:space:]]+.*${marker}([[:space:]]|$)"
}

apply_recovery_double_enter() {
  tmux send-keys -t "$PANE_ID" Enter
  sleep "$SLEEP_BETWEEN_ENTERS"
  tmux send-keys -t "$PANE_ID" Enter
}

apply_recovery_esc_plan_dismiss() {
  tmux send-keys -t "$PANE_ID" Escape
  sleep 2
  tmux send-keys -t "$PANE_ID" Enter
  sleep "$SLEEP_BETWEEN_ENTERS"
  tmux send-keys -t "$PANE_ID" Enter
}

generate_warmup_token() {
  # epoch_seconds + 4-5 decimal chars from /dev/urandom; only [0-9_].
  local rnd
  rnd="$(od -An -N2 -i /dev/urandom 2>/dev/null | tr -d ' ' || echo "$$")"
  printf '%s_%s' "$(date +%s)" "$rnd"
}

# state variables (shared between phases via finalize)
recovery_method="none"
recovery_count=0
plan_dialog="false"
first_attempt="unknown"

finalize() {
  local ack_status="$1"
  local exit_code="$2"
  local t1
  t1=$(date +%s)
  local elapsed=$((t1 - T0))
  printf 'DISPATCH_TO_PANE_RESULT pane=%s bytes=%s phase=%s warmup_seconds=%s first_attempt=%s recovery=%s rcount=%s plan_dialog=%s ack=%s seconds=%s\n' \
    "$PANE_ID" "$LEN" "$CURRENT_PHASE" "$WARMUP_ELAPSED" "$first_attempt" "$recovery_method" "$recovery_count" "$plan_dialog" "$ack_status" "$elapsed" >&2
  exit "$exit_code"
}

# === do_prewarm: optional Phase 0 ===
# Sends a tiny warmup payload with its own ACK marker and waits up to
# WARMUP_TIMEOUT_SEC for the warmup ACK to appear. On success, resets per-phase
# state for the main phase and returns to caller. On failure, calls finalize
# with the appropriate exit code (4 = tmux mechanical, 5 = warmup logical).
do_prewarm() {
  CURRENT_PHASE="warmup"
  WARMUP_T0=$(date +%s)
  local token warmup_payload warmup_marker
  token="$(generate_warmup_token)"
  warmup_marker="PREWARM_ACK_${token}"
  warmup_payload="DISPATCH_TO_PANE_PREWARM_PING ${token}: please reply EXACTLY with: ${warmup_marker}"

  # Phase 0a: chunked literal send (exit 4 phase=warmup on tmux failure)
  local wlen=${#warmup_payload}
  local woff=0
  while (( woff < wlen )); do
    if ! tmux send-keys -t "$PANE_ID" -l -- "${warmup_payload:woff:CHUNK_SIZE}" 2>/dev/null; then
      first_attempt="ERROR"
      finalize "UNKNOWN" 4
    fi
    woff=$((woff + CHUNK_SIZE))
  done
  tmux send-keys -t "$PANE_ID" Enter || { first_attempt="ERROR"; finalize "UNKNOWN" 4; }
  sleep "$SLEEP_BETWEEN_ENTERS"
  tmux send-keys -t "$PANE_ID" Enter || { first_attempt="ERROR"; finalize "UNKNOWN" 4; }

  # Phase 0b: post-dispatch state verification
  sleep "$INITIAL_VERIFY_SLEEP"
  local state
  state="$(capture_state)"
  if echo "$state" | grep -Eq "^•[[:space:]]+.*${warmup_marker}([[:space:]]|$)"; then
    first_attempt="YES"
  elif echo "$state" | grep -q "^• Working\|◦ Working"; then
    first_attempt="YES_WORKING"
  elif echo "$state" | grep -q "Create a plan?"; then
    first_attempt="NO"
    plan_dialog="true"
    recovery_method="esc_plan_dismiss"
    apply_recovery_esc_plan_dismiss
    recovery_count=$((recovery_count + 1))
  else
    first_attempt="NO"
    recovery_method="double_enter"
    apply_recovery_double_enter
    recovery_count=$((recovery_count + 1))
  fi

  # Phase 0c: warmup ACK wait
  local wdeadline=$((SECONDS + WARMUP_TIMEOUT_SEC))
  while (( SECONDS < wdeadline )); do
    if ack_present_in_full "$warmup_marker"; then
      WARMUP_ELAPSED=$(( $(date +%s) - WARMUP_T0 ))
      # reset per-phase state for the main phase; preserve WARMUP_ELAPSED for finalize.
      CURRENT_PHASE="main"
      recovery_method="none"
      recovery_count=0
      plan_dialog="false"
      first_attempt="unknown"
      return 0
    fi
    sleep 3
    local s2
    s2="$(capture_state)"
    if echo "$s2" | grep -q "Create a plan?" && (( recovery_count < MAX_RECOVERY )); then
      plan_dialog="true"
      if [[ "$recovery_method" == "none" ]]; then
        recovery_method="esc_plan_dismiss"
      else
        recovery_method="${recovery_method}_extra"
      fi
      apply_recovery_esc_plan_dismiss
      recovery_count=$((recovery_count + 1))
      sleep 2
      continue
    fi
  done

  # warmup timeout — logical failure (tmux worked, Codex did not produce ACK)
  WARMUP_ELAPSED=$(( $(date +%s) - WARMUP_T0 ))
  finalize "NO" 5
}

# === optionally run Phase 0 pre-warm ===
# Skipped when WARMUP_TIMEOUT_SEC=0 (equivalent to --no-prewarm).
if [[ "$PREWARM_ENABLED" == "1" && "${WARMUP_TIMEOUT_SEC:-0}" -gt 0 ]]; then
  do_prewarm
fi

# === Phase 1: chunked literal send (main payload) ===
# Use "--" end-of-options separator so chunk content starting with "-"
# (e.g. "-0xDEADBEEF" or "-1.5e-3" mid-payload) is not mis-parsed as a flag.
OFF=0
while (( OFF < CHAR_LEN )); do
  if ! tmux send-keys -t "$PANE_ID" -l -- "${PAYLOAD:OFF:CHUNK_SIZE}" 2>/dev/null; then
    first_attempt="ERROR"
    finalize "UNKNOWN" 4
  fi
  OFF=$((OFF + CHUNK_SIZE))
done
tmux send-keys -t "$PANE_ID" Enter || { first_attempt="ERROR"; finalize "UNKNOWN" 4; }
sleep "$SLEEP_BETWEEN_ENTERS"
tmux send-keys -t "$PANE_ID" Enter || { first_attempt="ERROR"; finalize "UNKNOWN" 4; }

# === Phase 2: post-dispatch state verification (main payload) ===
sleep "$INITIAL_VERIFY_SLEEP"
state="$(capture_state)"

# Initial classification (resets per-phase state — pre-warm path already reset
# these on its successful return; this is the canonical entry for main phase).
recovery_method="none"
recovery_count=0
plan_dialog="false"
first_attempt="unknown"

if [[ -n "$ACK_MARKER" ]] && echo "$state" | grep -Eq "^•[[:space:]]+.*${ACK_MARKER}([[:space:]]|$)"; then
  first_attempt="YES"
elif echo "$state" | grep -q "^• Working\|◦ Working"; then
  first_attempt="YES_WORKING"
elif echo "$state" | grep -q "Create a plan?"; then
  first_attempt="NO"
  plan_dialog="true"
  recovery_method="esc_plan_dismiss"
  apply_recovery_esc_plan_dismiss
  recovery_count=$((recovery_count + 1))
else
  first_attempt="NO"
  recovery_method="double_enter"
  apply_recovery_double_enter
  recovery_count=$((recovery_count + 1))
fi

# === Phase 3: ACK wait OR submission-only verification (main) ===
if [[ -z "$ACK_MARKER" ]]; then
  # Track Q §4.2 polish: re-capture after recovery, only OK if Working visible.
  if [[ "$first_attempt" == "YES" || "$first_attempt" == "YES_WORKING" ]]; then
    finalize "UNKNOWN" 0
  fi
  sleep "$SLEEP_BETWEEN_ENTERS"
  s_post="$(capture_state)"
  if echo "$s_post" | grep -q "^• Working\|◦ Working"; then
    finalize "UNKNOWN" 0
  fi
  if (( recovery_count >= MAX_RECOVERY )); then
    finalize "UNKNOWN" 3
  fi
  finalize "UNKNOWN" 0
fi

deadline=$((SECONDS + TIMEOUT_SEC))
working_seen="false"

while (( SECONDS < deadline )); do
  if ack_present_in_full "$ACK_MARKER"; then
    finalize "YES" 0
  fi
  sleep 3
  s2="$(capture_state)"
  if echo "$s2" | grep -q "^• Working\|◦ Working"; then
    working_seen="true"
  fi
  if echo "$s2" | grep -q "Create a plan?" && (( recovery_count < MAX_RECOVERY )); then
    plan_dialog="true"
    # Track Q §4.1 polish: explicit branch on initial recovery_method instead of
    # ${var:-default} which never substitutes for non-empty values like "none".
    if [[ "$recovery_method" == "none" ]]; then
      recovery_method="esc_plan_dismiss"
    else
      recovery_method="${recovery_method}_extra"
    fi
    apply_recovery_esc_plan_dismiss
    recovery_count=$((recovery_count + 1))
    sleep 2
    continue
  fi
done

# Timeout reached (main phase)
if [[ "$working_seen" == "true" ]]; then
  finalize "NO" 1   # submission OK but ACK timeout
fi

if (( recovery_count >= MAX_RECOVERY )); then
  finalize "NO" 3   # recovery exhausted
fi

finalize "NO" 1
