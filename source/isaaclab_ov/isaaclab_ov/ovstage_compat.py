# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVStage clone compatibility without importing the native runtime."""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

try:
    OVSTAGE_VERSION = Version(version("ovstage"))
except PackageNotFoundError:
    OVSTAGE_VERSION = None

# The 0.4 release line needs CPU transform propagation before cloning, including development builds.
OVSTAGE_REQUIRES_CPU_HIERARCHY_BEFORE_CLONE = OVSTAGE_VERSION is not None and OVSTAGE_VERSION.release[:2] == (0, 4)
