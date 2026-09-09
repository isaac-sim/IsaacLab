# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version compatibility between the public OVRTX 0.4 API and OVRTX 0.5 and later.

OVRTX 0.4 keys ``frame.render_vars`` by render-var source name (``"LdrColor"``), while
0.5 keys it by the authored RenderVar prim path (``"/Render/Vars/LdrColor"``). The
installed version cannot change while the process runs, so the key form is resolved once
at import and published as :data:`RENDER_VAR_FRAME_KEYS`; per-frame code indexes that
mapping instead of re-checking the version.

The public extras stay pinned to ``ovrtx==0.4.1.364340``; a missing or unparsable install
keeps the 0.4 key form.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Mapping
from types import MappingProxyType

from packaging.version import InvalidVersion, Version

from .ovrtx_usd import render_var_prim_paths_by_source

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


def build_render_var_frame_keys(version: Version | None) -> Mapping[str, str]:
    """Return the ``frame.render_vars`` key for every render-var source under ``version``.

    Args:
        version: OVRTX version the keys are built for, or ``None`` when OVRTX is unavailable.

    Returns:
        Read-only mapping of render-var source name to frame key: the source name itself on
        OVRTX 0.4, or the authored RenderVar prim path on OVRTX 0.5 and later.
    """
    prim_paths = render_var_prim_paths_by_source()
    if uses_prim_path_render_vars(version):
        return MappingProxyType(dict(prim_paths))
    return MappingProxyType({source: source for source in prim_paths})


OVRTX_VERSION: Version | None = detect_ovrtx_version()
"""Installed OVRTX version, or ``None`` when it is unavailable or unparsable."""

RENDER_VAR_FRAME_KEYS: Mapping[str, str] = build_render_var_frame_keys(OVRTX_VERSION)
"""Maps render-var source name to its ``frame.render_vars`` key for the installed OVRTX."""
