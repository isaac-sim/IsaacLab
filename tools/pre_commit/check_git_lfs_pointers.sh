#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Pre-commit hook: ensure Git LFS attribute rules match index storage.
#
# Paths with filter=lfs in .gitattributes must be staged as LFS pointers, not plain
# blobs. When many paths are checked (e.g. pre-commit run --all-files), falls back to
# git lfs fsck --pointers.

set -euo pipefail

# Above this count, per-file index checks are slower than a single fsck over HEAD/index.
readonly FSCK_FILE_THRESHOLD=200

# --- Prerequisites -----------------------------------------------------------
# Git LFS must be on PATH; without it, filter=clean/smudge cannot run on commit.
if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is not installed; install it to work with LFS-tracked files." >&2
    exit 1
fi

# --- Helpers: read .gitattributes and index blobs ----------------------------

# Return the "filter" attribute value for a path (e.g. "lfs" or "unspecified").
lfs_filter_attr() {
    local path="$1"
    git check-attr filter -- "${path}" 2>/dev/null | sed -n 's/.*filter: //p' | tr -d ' \r'
}

# True when the staged blob (":path") is a well-formed LFS pointer file.
index_blob_is_valid_lfs_pointer() {
    local path="$1"
    git show ":${path}" 2>/dev/null | git lfs pointer --check --stdin >/dev/null 2>&1
}

# True when the staged blob starts with the LFS pointer spec line.
# Used for the reverse check only: "pointer --check" accepts empty files, which
# would false-positive on .gitkeep and other zero-byte paths.
index_blob_has_lfs_pointer_header() {
    local path="$1"
    git show ":${path}" 2>/dev/null | head -n1 | grep -qxF 'version https://git-lfs.github.com/spec/v1'
}

# --- User-facing errors ------------------------------------------------------
print_fix_hint() {
    echo >&2
    echo "For LFS-tracked files, re-add after Git LFS is installed:" >&2
    echo "  git rm --cached <file> && git add <file>" >&2
}

# --- Full-repository check (pre-commit run --all-files / many paths) ---------
# Verifies every path that should use LFS is stored as a pointer in HEAD/index.
run_fsck() {
    git lfs fsck --pointers
}

# --- Incremental check (git commit / small diffs) ------------------------------
# pre-commit passes staged/changed paths as "$@". For each path in the index:
#   1. filter=lfs  -> blob must be a valid LFS pointer
#   2. not filter=lfs but LFS header -> misconfigured .gitattributes
check_paths() {
    local path filter_attr
    local -a error_messages=()

    for path in "$@"; do
        filter_attr="$(lfs_filter_attr "${path}")"
        # Skip paths not present in the index (e.g. deleted-only in this commit).
        if ! git show ":${path}" >/dev/null 2>&1; then
            continue
        fi

        if [[ "${filter_attr}" == "lfs" ]]; then
            if ! index_blob_is_valid_lfs_pointer "${path}"; then
                error_messages+=(
                    "  ${path}: has filter=lfs but is staged as a regular Git blob (not an LFS pointer)"
                )
            fi
        elif index_blob_has_lfs_pointer_header "${path}"; then
            error_messages+=(
                "  ${path}: is an LFS pointer but is not tracked by filter=lfs in .gitattributes"
            )
        fi
    done

    if [[ "${#error_messages[@]}" -ne 0 ]]; then
        echo "Git LFS pointer check failed:" >&2
        printf '%s\n' "${error_messages[@]}" >&2
        print_fix_hint
        return 1
    fi
    return 0
}

# --- Entry point -------------------------------------------------------------
# No args or a large file list -> fsck; otherwise check only the given paths.
main() {
    if [[ "$#" -eq 0 ]] || [[ "$#" -gt "${FSCK_FILE_THRESHOLD}" ]]; then
        run_fsck
        return
    fi
    check_paths "$@"
}

main "$@"
