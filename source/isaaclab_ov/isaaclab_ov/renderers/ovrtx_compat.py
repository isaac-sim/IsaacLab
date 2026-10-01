# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version compatibility between the public OVRTX 0.4 API and OVRTX 0.5 and later.

OVRTX 0.4 keys ``frame.render_vars`` by render-var source name (``"LdrColor"``), while
0.5 keys it by the authored RenderVar prim path (``"/RenderCamera_0/Vars/LdrColor"``).
The renderer resolves source names to frame keys when reading each camera's output.

Missing or invalid version metadata selects source-name render-var keys.
"""

from __future__ import annotations

import importlib.metadata
import logging

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# First OVRTX version that keys ``frame.render_vars`` by RenderVar prim path.
_PRIM_PATH_RENDER_VARS_VERSION = Version("0.5")


def detect_ovrtx_version() -> Version | None:
    """Return the installed ``ovrtx`` version.

    Read from distribution metadata rather than ``ovrtx.__version__`` so it does not
    require importing the runtime. An unparsable version is logged and reported as
    missing, which keeps the OVRTX 0.4 behavior.

    Returns:
        The installed version, or ``None`` when ``ovrtx`` is absent or its version string
        cannot be parsed.
    """
    try:
        raw = importlib.metadata.version("ovrtx")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return Version(raw)
    except InvalidVersion:
        logger.warning("Could not parse ovrtx version %r; assuming the OVRTX 0.4 render-var API.", raw)
        return None


def uses_prim_path_render_vars(version: Version | None) -> bool:
    """Return whether ``version`` keys ``frame.render_vars`` by RenderVar prim path.

    Args:
        version: OVRTX version to classify, or ``None`` when OVRTX is unavailable.

    Returns:
        Whether ``version`` is OVRTX 0.5 or newer.
    """
    return version is not None and version >= _PRIM_PATH_RENDER_VARS_VERSION


OVRTX_VERSION: Version | None = detect_ovrtx_version()
"""Installed OVRTX version, or ``None`` when it is unavailable or unparsable."""
