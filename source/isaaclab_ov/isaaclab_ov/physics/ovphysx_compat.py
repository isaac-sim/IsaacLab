# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version compatibility between the public OVPhysX 0.5.11 API and OVPhysX 0.6 and later.

OVPhysX 0.5.11 warms GPU state with ``warmup_gpu()`` and tears the runtime down
with ``release()``, while 0.6 replaces those entry points with ``warmup()`` and
``destroy()``. The installed version cannot change while the process runs, so
the entry-point names are resolved once at import and published as
:data:`OVPHYSX_LIFECYCLE_ENTRY_POINTS`.

The public extras stay pinned to ``ovphysx==0.5.11``; a missing or unparsable
install keeps the 0.5.11 entry points.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Mapping
from types import MappingProxyType

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# First OVPhysX version that uses the current lifecycle entry points.
_CURRENT_LIFECYCLE_VERSION = Version("0.6")


def detect_ovphysx_version() -> Version | None:
    """Return the installed ``ovphysx`` version.

    Read from distribution metadata rather than importing ``ovphysx`` so the
    runtime is still bootstrapped only by the physics manager. An unparsable
    version is logged and reported as missing, which keeps the OVPhysX 0.5.11
    behavior.

    Returns:
        The installed version, or ``None`` when ``ovphysx`` is absent or its
        version string cannot be parsed.
    """
    try:
        raw = importlib.metadata.version("ovphysx")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return Version(raw)
    except InvalidVersion:
        logger.warning("Could not parse ovphysx version %r; assuming the OVPhysX 0.5.11 lifecycle API.", raw)
        return None


def uses_current_lifecycle_api(version: Version | None) -> bool:
    """Return whether ``version`` uses the OVPhysX 0.6 lifecycle API.

    Args:
        version: OVPhysX version to classify, or ``None`` when OVPhysX is unavailable.

    Returns:
        Whether ``version`` is OVPhysX 0.6 or newer.
    """
    return version is not None and version >= _CURRENT_LIFECYCLE_VERSION


def build_lifecycle_entry_points(version: Version | None) -> Mapping[str, str]:
    """Return the lifecycle entry-point names for ``version``.

    Args:
        version: OVPhysX version the names are built for, or ``None`` when
            OVPhysX is unavailable.

    Returns:
        Read-only mapping from lifecycle operation to runtime method name.
    """
    if uses_current_lifecycle_api(version):
        entry_points = {"warmup": "warmup", "destroy": "destroy"}
    else:
        entry_points = {"warmup": "warmup_gpu", "destroy": "release"}
    return MappingProxyType(entry_points)


OVPHYSX_VERSION: Version | None = detect_ovphysx_version()
"""Installed OVPhysX version, or ``None`` when it is unavailable or unparsable."""

OVPHYSX_LIFECYCLE_ENTRY_POINTS: Mapping[str, str] = build_lifecycle_entry_points(OVPHYSX_VERSION)
"""Maps lifecycle operations to method names for the installed OVPhysX version."""
