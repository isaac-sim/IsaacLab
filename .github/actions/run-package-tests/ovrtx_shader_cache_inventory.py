# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report the OVRTX shader cache inventory and expose per-tree counts for save gating.

Sizes alone cannot answer whether a tree was populated - an empty directory still
measures around 1 MB - so the per-tree file counts written to ``GITHUB_OUTPUT``
(``kit-files``, ``kit-bytes``, ``kitless-files``, ``kitless-bytes``) are what the
caller gates its writeback on.

Usage:
    python3 ovrtx_shader_cache_inventory.py <host_dir> <label>
"""

from __future__ import annotations

import os
import sys

TREES = ("kit", "kitless")


def _dir_stats(path: str) -> tuple[int, int]:
    """Return (file_count, total_bytes) for a directory tree (0, 0 if missing)."""
    if not os.path.isdir(path):
        return 0, 0
    count = 0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fname))
                count += 1
            except OSError:
                pass
    return count, total


def _emit_outputs(stats: dict[str, tuple[int, int]]) -> None:
    """Publish per-tree counts and byte totals as step outputs, when running in a step."""
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a", encoding="utf-8") as handle:
        for tree, (count, total) in stats.items():
            handle.write(f"{tree}-files={count}\n")
            handle.write(f"{tree}-bytes={total}\n")


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <host_dir> <label>", file=sys.stderr)
        return 1

    host_dir = sys.argv[1]
    label = sys.argv[2]

    stats = {tree: _dir_stats(os.path.join(host_dir, tree)) for tree in TREES}
    megabytes = {tree: total / (1024 * 1024) for tree, (_, total) in stats.items()}

    print(f"{label}:")
    for tree in TREES:
        count, _ = stats[tree]
        print(f"  {tree + '/':10} {count} file(s), {megabytes[tree]:.1f} MB")
    print(f"  {'total':10} {sum(megabytes.values()):.1f} MB")

    _emit_outputs(stats)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(
                f"🔵 OVRTX shader cache ({label}): "
                f"kit={megabytes['kit']:.0f} MB, kitless={megabytes['kitless']:.0f} MB, "
                f"total={sum(megabytes.values()):.0f} MB\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
