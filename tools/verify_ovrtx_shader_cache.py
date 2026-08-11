# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that the OVRTX shader cache directories are mounted and writable.

Run inside the test container before tests start, so a bad mount surfaces here
rather than as unexplained slow shader compilation mid-test.

Both trees are checked. kit/ arrives as a nested bind mount that has to land on
top of the enclosing kit/cache tmpdir mount, so it is the easier of the two to
get silently wrong: when it does, the restore step still reports a hit and the
only symptom is a rendering job that takes several minutes longer.

Exit 0 on success; non-zero on any misconfiguration.
"""

from __future__ import annotations

import os
import sys

# Each tree is named by the env var carrying its container path; see the mount
# layout documented in .github/actions/run-tests/run_tests.sh. A tree whose
# variable is unset is not mounted for this job and is skipped.
CACHE_PATH_ENVS = {
    "kitless": "OVRTX_SHADER_CACHE_PATH",
    "kit": "OVRTX_KIT_SHADER_CACHE_PATH",
}


def _dir_stats(path: str) -> tuple[int, float]:
    """Return ``(file_count, megabytes)`` for the tree rooted at ``path``."""
    total_bytes = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            try:
                total_bytes += os.path.getsize(os.path.join(dirpath, fname))
                file_count += 1
            except OSError:
                pass
    return file_count, total_bytes / (1024 * 1024)


def check_tree(tree: str, cache_path: str) -> bool:
    """Report whether ``cache_path`` exists and is writable, logging what it holds."""
    if not os.path.isdir(cache_path):
        print(
            f"[verify_ovrtx_shader_cache] {tree}/ cache directory does not exist: {cache_path!r}",
            file=sys.stderr,
        )
        return False

    probe = os.path.join(cache_path, ".ovrtx_cache_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        print(
            f"[verify_ovrtx_shader_cache] {tree}/ cache is not writable: {cache_path!r} - {exc}",
            file=sys.stderr,
        )
        return False

    # Report size of any existing cache content so growth is visible in logs.
    file_count, total_mb = _dir_stats(cache_path)
    print(
        f"[verify_ovrtx_shader_cache] {tree}/ cache OK - path={cache_path!r}, {file_count} file(s), {total_mb:.1f} MB"
    )
    return True


def main() -> int:
    mounted = {tree: os.environ[env] for tree, env in CACHE_PATH_ENVS.items() if os.environ.get(env)}
    if not mounted:
        print(
            "[verify_ovrtx_shader_cache] no OVRTX shader cache paths are set; skipping.",
            file=sys.stderr,
        )
        return 0

    # A list, not a generator: every tree gets checked and reported even after
    # one fails, so a single run shows every broken mount rather than the first.
    results = [check_tree(tree, path) for tree, path in sorted(mounted.items())]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
