#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reports the state of the OVRTX shader cache, either side of the test run.
#
# Usage: report.sh restore|growth
#
#   restore  prepares both trees, checks the mount is writable, prints the
#            per-tree hit/miss and records the baseline the growth pass compares
#            against
#   growth   prints how far the run compiled beyond what was restored, and
#            emits the per-tree file counts the save gates read
#
# OVRTX_CACHE_HIT and OVRTX_CACHE_MB_BEFORE are how the restore pass hands the
# baseline to the growth pass. They are written to $GITHUB_ENV because the two
# passes are separate invocations of this action; they are internal to it and no
# caller should read or set them.

set -euo pipefail

mode="${1:?usage: report.sh restore|growth}"
: "${HOST_DIR:?HOST_DIR is required}"

# Files, not bytes: an empty tree still measures ~1 MB, so only a per-tree count
# separates a populated tree from one the tests never filled. tr strips wc's
# padding - a stray space would make the '0' comparisons in the save gates miss
# and publish an empty tree.
count_files() {
  find "$1" -type f 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

case "$mode" in
  restore)
    mkdir -p "$HOST_DIR/kit" "$HOST_DIR/kitless"
    if [ ! -w "$HOST_DIR" ]; then
      echo "::error::OVRTX shader cache directory is not writable: $HOST_DIR"
      exit 1
    fi

    # cache-matched-key is authoritative. Inferring a hit from directory size is
    # not: an empty directory still measures as ~1 MB.
    if [ -z "${KIT_MATCHED_KEY:-}" ]; then
      echo "::warning::OVRTX kit shader cache miss - no entry in collection ${KIT_COLLECTION:-}"
    else
      echo "OVRTX kit shader cache hit: ${KIT_MATCHED_KEY}"
    fi
    if [ -z "${KITLESS_MATCHED_KEY:-}" ]; then
      echo "::warning::OVRTX kitless shader cache miss - no entry in collection ${KITLESS_COLLECTION:-}"
    else
      echo "OVRTX kitless shader cache hit: ${KITLESS_MATCHED_KEY}"
    fi

    if [ -z "${KIT_MATCHED_KEY:-}" ] && [ -z "${KITLESS_MATCHED_KEY:-}" ]; then
      echo "OVRTX_CACHE_HIT=miss" >> "$GITHUB_ENV"
    else
      echo "OVRTX_CACHE_HIT=${KIT_MATCHED_KEY:-miss}/${KITLESS_MATCHED_KEY:-miss}" >> "$GITHUB_ENV"
    fi
    echo "OVRTX_CACHE_MB_BEFORE=$(du -sm "$HOST_DIR" | cut -f1)" >> "$GITHUB_ENV"
    ;;

  growth)
    if [ ! -d "$HOST_DIR" ]; then
      echo "OVRTX shader cache directory missing; nothing to report"
      echo "kit-files=0" >> "$GITHUB_OUTPUT"
      echo "kitless-files=0" >> "$GITHUB_OUTPUT"
      exit 0
    fi

    before="${OVRTX_CACHE_MB_BEFORE:-0}"
    after="$(du -sm "$HOST_DIR" | cut -f1)"
    grew=$(( after - before ))

    if [ "${OVRTX_CACHE_HIT:-miss}" = "miss" ]; then
      verdict="cold run, nothing restored"
    elif [ "$grew" -le 0 ]; then
      verdict="fully covered"
    else
      verdict="compiled $(( grew * 100 / (before > 0 ? before : 1) ))% beyond the restored cache"
    fi

    echo "OVRTX shader cache: restored ${before} MB, ended at ${after} MB (+${grew} MB) - ${verdict}"
    echo "🔵 OVRTX shader cache: ${before} -> ${after} MB (+${grew}) - ${verdict}" >> "$GITHUB_STEP_SUMMARY"

    for tree in kit kitless; do
      files="$(count_files "$HOST_DIR/$tree")"
      echo "  ${tree}/ ${files:-0} file(s)"
      echo "${tree}-files=${files:-0}" >> "$GITHUB_OUTPUT"
    done
    ;;

  *)
    echo "::error::unknown report mode: ${mode}"
    exit 1
    ;;
esac
