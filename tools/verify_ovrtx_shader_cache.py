# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that the OVRTX kitless shader cache is mounted and writable.

Run inside the test container before tests start to catch mount problems early
rather than discovering them as slow shader compilation mid-test.

The kit/ sub-tree is validated separately by the run_tests.sh mount setup
(the Docker nested bind mount either works or the container fails to start);
this script only checks the kitless path exposed via OVRTX_SHADER_CACHE_PATH
because that variable is what OVRTXRenderer reads at Renderer-creation time.

Exit 0 on success; non-zero on any misconfiguration.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    cache_path = os.environ.get("OVRTX_SHADER_CACHE_PATH")
    if not cache_path:
        print(
            "[verify_ovrtx_shader_cache] OVRTX_SHADER_CACHE_PATH is not set; skipping.",
            file=sys.stderr,
        )
        return 0

    if not os.path.isdir(cache_path):
        print(
            f"[verify_ovrtx_shader_cache] OVRTX_SHADER_CACHE_PATH directory does not exist: {cache_path!r}",
            file=sys.stderr,
        )
        return 1

    probe = os.path.join(cache_path, ".ovrtx_cache_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        print(
            f"[verify_ovrtx_shader_cache] OVRTX_SHADER_CACHE_PATH is not writable: {cache_path!r} — {exc}",
            file=sys.stderr,
        )
        return 1

    # Report size of any existing cache content so growth is visible in logs.
    total_bytes = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(cache_path):
        for fname in filenames:
            try:
                total_bytes += os.path.getsize(os.path.join(dirpath, fname))
                file_count += 1
            except OSError:
                pass
    total_mb = total_bytes / (1024 * 1024)

    print(
        f"[verify_ovrtx_shader_cache] kitless cache OK — path={cache_path!r}, {file_count} file(s), {total_mb:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
