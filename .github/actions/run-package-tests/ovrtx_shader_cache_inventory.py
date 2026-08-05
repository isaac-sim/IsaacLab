# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report the OVRTX shader cache inventory to the GitHub step summary.

The host directory has two sub-directories:
  kit/      — nv_shadercache from Kit / AppLauncher-based rendering
  kitless/  — nv_shadercache from standalone OVRTXRenderer

Usage:
    python3 ovrtx_shader_cache_inventory.py <host_dir> <label>
"""

from __future__ import annotations

import os
import sys


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


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <host_dir> <label>", file=sys.stderr)
        return 1

    host_dir = sys.argv[1]
    label = sys.argv[2]

    kit_dir = os.path.join(host_dir, "kit")
    kitless_dir = os.path.join(host_dir, "kitless")

    kit_count, kit_bytes = _dir_stats(kit_dir)
    kitless_count, kitless_bytes = _dir_stats(kitless_dir)

    kit_mb = kit_bytes / (1024 * 1024)
    kitless_mb = kitless_bytes / (1024 * 1024)
    total_mb = kit_mb + kitless_mb

    print(f"{label}:")
    print(f"  kit/      {kit_count} file(s), {kit_mb:.1f} MB")
    print(f"  kitless/  {kitless_count} file(s), {kitless_mb:.1f} MB")
    print(f"  total     {total_mb:.1f} MB")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(
                f"🔵 OVRTX shader cache ({label}): "
                f"kit={kit_mb:.0f} MB, kitless={kitless_mb:.0f} MB, total={total_mb:.0f} MB\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
