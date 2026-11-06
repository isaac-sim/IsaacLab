# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assert that Warp resolved its kernel cache to the directory CI mounted.

Run inside the test container when ``WARP_CACHE_PATH`` is set. A Warp that
silently ignores the mount is indistinguishable from a working cache that never
gets a hit, so this fails the job loudly instead.

Lives under ``tools/`` rather than beside the action because ``.dockerignore``
excludes ``.github/``, so a copy there would be missing from any container that
is not bind-mounted.
"""

import os
import re

import warp as wp

wp.init()

expected = os.path.realpath(os.environ["WARP_CACHE_PATH"])
actual = os.path.realpath(wp.config.kernel_cache_dir)

print(f"Warp version: {wp.config.version}")
print(f"Warp kernel cache directory: {actual}")

if os.path.commonpath([actual, expected]) != expected:
    raise SystemExit(f"Warp cache path mismatch: expected {expected}, got {actual}")

# The cache key is scoped on the warp-lang version pinned in the lockfile, and
# Warp namespaces its cache dir by its own version. If the two disagree the key
# names a namespace nothing ever writes, so every restore misses while looking
# like a working cache. Fail instead of degrading silently.
with open("uv.lock") as handle:
    pinned = re.search(r'name = "warp-lang"\nversion = "([^"]+)"', handle.read())

if pinned and pinned.group(1) != wp.config.version:
    raise SystemExit(
        f"Warp version mismatch: uv.lock pins {pinned.group(1)} but the container runs"
        f" {wp.config.version}; the cache key would never match what Warp writes"
    )
