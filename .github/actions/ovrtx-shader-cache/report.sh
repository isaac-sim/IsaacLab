#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reports the state of the OVRTX shader cache, either side of the test run.
#
# Usage: report.sh restore|growth
#
#   restore  prepares the requested tree(s), checks the mount is writable, prints
#            the per-tree hit/miss and records the baseline for the growth pass
#   growth   prints how far the run compiled beyond what was restored, and
#            emits the per-tree file counts and changed flags the save gates read
#
# TREES selects which tree(s) to process: 'kit', 'kitless' or 'both' (default).
# A tree the caller did not request is skipped entirely.
#
# The passes are separate invocations of this action, so the restore pass hands
# its baseline to the growth pass through OVRTX_CACHE_HIT, OVRTX_CACHE_MB_BEFORE
# and OVRTX_<TREE>_FINGERPRINT_BEFORE in $GITHUB_ENV. All are internal to this
# action.

set -euo pipefail

mode="${1:?usage: report.sh restore|growth}"
: "${HOST_DIR:?HOST_DIR is required}"

# 'kit', 'kitless' or 'both' (default); the tree(s) this job requested.
trees="${TREES:-both}"
requested() {
  [ "$trees" = "both" ] || [ "$trees" = "$1" ]
}

# Files, not bytes: an empty tree still measures ~1 MB. tr strips wc's padding,
# which would otherwise make the '0' comparisons in the save gates miss.
count_files() {
  find "$1" -type f 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

fingerprint_var() {
  printf 'OVRTX_%s_FINGERPRINT_BEFORE' "$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
}

# Digest of every file's path and contents in one tree, so the save pass can
# recognise a run that added nothing and skip publishing a duplicate snapshot.
# Contents rather than size or file count, because the driver rewrites blobs in
# place as well as appending them. Prints nothing when the tree is missing;
# callers read an empty digest as "not measured" and publish anyway.
fingerprint_tree() {
  [ -d "$1" ] || return 0
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum) |
    sha256sum | cut -d' ' -f1
}

case "$mode" in
  restore)
    hit=""
    if requested kit; then
      mkdir -p "$HOST_DIR/kit"
      # cache-matched-key is authoritative; directory size is not, since an empty
      # directory still measures as ~1 MB.
      if [ -z "${KIT_MATCHED_KEY:-}" ]; then
        echo "::warning::OVRTX kit shader cache miss - no entry in collection ${KIT_COLLECTION:-}"
      else
        echo "OVRTX kit shader cache hit: ${KIT_MATCHED_KEY}"
        hit="${hit}${KIT_MATCHED_KEY}/"
      fi
    fi
    if requested kitless; then
      mkdir -p "$HOST_DIR/kitless"
      if [ -z "${KITLESS_MATCHED_KEY:-}" ]; then
        echo "::warning::OVRTX kitless shader cache miss - no entry in collection ${KITLESS_COLLECTION:-}"
      else
        echo "OVRTX kitless shader cache hit: ${KITLESS_MATCHED_KEY}"
        hit="${hit}${KITLESS_MATCHED_KEY}/"
      fi
    fi
    if [ ! -w "$HOST_DIR" ]; then
      echo "::error::OVRTX shader cache directory is not writable: $HOST_DIR"
      exit 1
    fi

    echo "OVRTX_CACHE_HIT=${hit:-miss}" >> "$GITHUB_ENV"
    echo "OVRTX_CACHE_MB_BEFORE=$(du -sm "$HOST_DIR" | cut -f1)" >> "$GITHUB_ENV"

    # Only the warmer compares fingerprints, and hashing both trees is the one
    # part of this pass that scales with the restored snapshot.
    if [ "${PUBLISHES:-false}" = "true" ]; then
      for tree in kit kitless; do
        requested "$tree" || continue
        echo "$(fingerprint_var "$tree")=$(fingerprint_tree "$HOST_DIR/$tree")" >> "$GITHUB_ENV"
      done
    fi
    ;;

  growth)
    if [ ! -d "$HOST_DIR" ]; then
      echo "OVRTX shader cache directory missing; nothing to report"
      for tree in kit kitless; do
        requested "$tree" || continue
        echo "${tree}-files=0" >> "$GITHUB_OUTPUT"
        echo "${tree}-changed=false" >> "$GITHUB_OUTPUT"
      done
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
      requested "$tree" || continue
      files="$(count_files "$HOST_DIR/$tree")"
      files="${files:-0}"

      # A baseline this job never took - a consumer, or a save whose restore
      # pass was skipped - leaves the digest empty and publishes, so an
      # unmeasured tree costs a duplicate snapshot rather than a lost one.
      before_var="$(fingerprint_var "$tree")"
      before_fp="${!before_var:-}"
      if [ -n "$before_fp" ] && [ "$(fingerprint_tree "$HOST_DIR/$tree")" = "$before_fp" ]; then
        changed=false
        echo "  ${tree}/ ${files} file(s), unchanged since restore"
      else
        changed=true
        echo "  ${tree}/ ${files} file(s)"
      fi
      echo "${tree}-files=${files}" >> "$GITHUB_OUTPUT"
      echo "${tree}-changed=${changed}" >> "$GITHUB_OUTPUT"
    done
    ;;

  *)
    echo "::error::unknown report mode: ${mode}"
    exit 1
    ;;
esac
