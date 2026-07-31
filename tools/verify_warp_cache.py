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

import warp as wp

wp.init()

expected = os.path.realpath(os.environ["WARP_CACHE_PATH"])
actual = os.path.realpath(wp.config.kernel_cache_dir)

# The cache key is scoped on the warp-lang version pinned in pyproject.toml.
# Warp namespaces its cache dir by its own version, so if the runtime version
# differs the key points at a namespace nothing writes and every restore misses
# silently. Printing it makes that diagnosable from the job log.
print(f"Warp version: {wp.config.version}")
print(f"Warp kernel cache directory: {actual}")

if os.path.commonpath([actual, expected]) != expected:
    raise SystemExit(f"Warp cache path mismatch: expected {expected}, got {actual}")
