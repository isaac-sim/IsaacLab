#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Package the exact tracked and intentional untracked worktree for OSMO rsync.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaaclab_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
sync_root="${1:-$(dirname "${isaaclab_root}")/.osmo-yam-cable-routing-sync}"
cable_asset_dir="source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/assets"

if [[ ! -f "${isaaclab_root}/pyproject.toml" ]]; then
    echo "Isaac Lab worktree not found at ${isaaclab_root}" >&2
    exit 1
fi
if [[ -z "${sync_root}" || "${sync_root}" == "/" || "${sync_root}" == "${isaaclab_root}" ]]; then
    echo "Refusing unsafe synchronization directory: ${sync_root}" >&2
    exit 1
fi

mkdir -p "${sync_root}"
archive_tmp="$(mktemp --tmpdir="${sync_root}" source.tar.gz.tmp.XXXXXX)"
metadata_tmp="$(mktemp --tmpdir="${sync_root}" source.metadata.tmp.XXXXXX)"
manifest_tmp="$(mktemp --tmpdir="${sync_root}" source.manifest.tmp.XXXXXX)"
status_tmp="$(mktemp --tmpdir="${sync_root}" git-status.txt.tmp.XXXXXX)"
trap 'rm -f "${archive_tmp}" "${metadata_tmp}" "${manifest_tmp}" "${status_tmp}"' EXIT

# Include tracked files plus intentional untracked work while respecting the repository ignore rules.
# The bundled YAM and ManipulationNet snapshots are the sole exceptions: USD files are ignored
# repository-wide, so add only USD/USDAs/USDCs below the cable task's exact asset directory. This
# captures a dirty research worktree without uploading unrelated ignored artifacts such as logs,
# virtual environments, caches, or the local Isaac Sim symlink.
(
    cd "${isaaclab_root}"
    git ls-files --cached --others --exclude-standard -z
    if [[ -d "${cable_asset_dir}" ]]; then
        find "${cable_asset_dir}" -type f \( -name '*.usd' -o -name '*.usda' -o -name '*.usdc' \) -print0
    fi
) | sort -z -u > "${manifest_tmp}"

if [[ ! -s "${manifest_tmp}" ]]; then
    echo "Refusing to package an empty source manifest." >&2
    exit 2
fi

# A source overlay must never become an accidental credential transport. Refuse common secret-bearing
# names among untracked files. Tracked .env templates are intentionally allowed because they have already
# gone through repository review.
while IFS= read -r -d '' relative_path; do
    case "${relative_path}" in
        .env|.env.*|*/.env|*/.env.*|.netrc|*/.netrc|.npmrc|*/.npmrc|*.key|*.pem|*.p12|*.pfx)
            echo "Refusing to package possible secret file: ${relative_path}" >&2
            exit 4
            ;;
    esac
done < <(git -C "${isaaclab_root}" ls-files --others --exclude-standard -z)

(
    cd "${isaaclab_root}"
    tar --null --files-from="${manifest_tmp}" --create --gzip --file="${archive_tmp}"
)

archive_sha256="$(sha256sum "${archive_tmp}" | awk '{print $1}')"
manifest_count="$(tr -cd '\0' < "${manifest_tmp}" | wc -c)"
dirty_file_count="$(git -C "${isaaclab_root}" status --short --untracked-files=all | wc -l)"
git -C "${isaaclab_root}" status --short --branch --untracked-files=all > "${status_tmp}"
status_sha256="$(sha256sum "${status_tmp}" | awk '{print $1}')"
{
    printf 'packaged_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'branch=%s\n' "$(git -C "${isaaclab_root}" branch --show-current)"
    printf 'commit=%s\n' "$(git -C "${isaaclab_root}" rev-parse HEAD)"
    printf 'source_sha256=%s\n' "${archive_sha256}"
    printf 'manifest_file_count=%s\n' "${manifest_count}"
    printf 'dirty_file_count=%s\n' "${dirty_file_count}"
    # Keep this last so its presence also proves that the metadata transfer is complete.
    printf 'git_status_sha256=%s\n' "${status_sha256}"
} > "${metadata_tmp}"

# Publish atomically. The remote runner treats source.sha256 as the readiness marker and validates the
# archive before extracting it.
rm -f "${sync_root}/source.sha256"
mv "${archive_tmp}" "${sync_root}/source.tar.gz"
mv "${metadata_tmp}" "${sync_root}/source.metadata"
mv "${status_tmp}" "${sync_root}/git-status.txt"
printf '%s\n' "${archive_sha256}" > "${sync_root}/source.sha256.tmp"
mv "${sync_root}/source.sha256.tmp" "${sync_root}/source.sha256"

printf '%s\n' "${sync_root}"
