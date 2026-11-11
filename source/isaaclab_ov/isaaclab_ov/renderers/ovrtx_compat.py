# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX format selection without importing the optional native runtime."""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

try:
    OVRTX_VERSION = Version(version("ovrtx"))
except PackageNotFoundError:
    # Documentation mocks the native runtime without installing its distribution.
    OVRTX_VERSION = None

OVRTX_USES_NAMED_SEGMENTATION_MAPS = OVRTX_VERSION is not None and OVRTX_VERSION.release[:2] >= (0, 7)
