# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version compatibility for OVPhysX lifecycle, tasks, schemas, and scene reads.

OVPhysX 0.5.11 warms GPU state with ``warmup_gpu()`` and tears the runtime down
with ``release()``, while 0.6 replaces those entry points with ``warmup()`` and
``destroy()``. The installed version cannot change while the process runs, so
the entry-point names are resolved once at import and published as
:data:`OVPHYSX_LIFECYCLE_ENTRY_POINTS`.

Missing or invalid version metadata selects the 0.5 lifecycle API and
reversed-joint correction.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# First OVPhysX release line that uses the current lifecycle entry points.
_CURRENT_LIFECYCLE_RELEASE = (0, 6)


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
        logger.warning("Could not parse ovphysx version %r; assuming OVPhysX 0.5.11 compatibility behavior.", raw)
        return None


def uses_current_lifecycle_api(version: Version | None) -> bool:
    """Return whether ``version`` uses the OVPhysX 0.6 lifecycle API.

    Args:
        version: OVPhysX version to classify, or ``None`` when OVPhysX is unavailable.

    Returns:
        Whether ``version`` is OVPhysX 0.6 or newer.
    """
    return version is not None and version.release[:2] >= _CURRENT_LIFECYCLE_RELEASE


def requires_legacy_joint_sign_correction(version: Version | None) -> bool:
    """Return whether dynamics tensors need the pre-0.6 reversed-joint correction.

    Args:
        version: Installed OVPhysX version, or ``None`` when unavailable or unparsable.

    Returns:
        Whether to preserve the legacy correction. The 0.6 release line, including
        development builds, already returns tensors in the public joint basis.
    """
    return version is None or version.release[:2] < (0, 6)


def uses_task_api(version: Version | None) -> bool:
    """Select the 0.7 contracts, including prereleases, without importing a native runtime."""
    return version is not None and version.release[:2] >= (0, 7)


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

OVPHYSX_USES_TASKS: bool = uses_task_api(OVPHYSX_VERSION)
"""Whether this runtime uses 0.7 task receipts and external PhysX schemas (including development builds)."""


def wait_for_operation(physx: Any, operation: Any) -> None:
    """Collect a clone or legacy reset operation exactly once."""
    if OVPHYSX_USES_TASKS:
        physx.wait_task(operation)
    else:
        physx.wait_op(operation)


def finish_stage_operation(physx: Any, operation: Any) -> None:
    """Collect an attach/update receipt; older runtimes already complete these calls synchronously."""
    if OVPHYSX_USES_TASKS:
        physx.wait_task(operation)


def detach_stage(physx: Any) -> None:
    """Complete stage teardown before its owner releases the OVStage."""
    operation = physx.detach_ovstage() if OVPHYSX_USES_TASKS else physx.reset_stage()
    wait_for_operation(physx, operation)


def physx_schema_paths(ovphysx: Any) -> list[Path]:
    """Locate the selected runtime's schemas without importing the external USD plugin provider."""
    if not OVPHYSX_USES_TASKS:
        return list(ovphysx.codeless_schema_paths())
    package = importlib.metadata.distribution("physx-usd-schemas")
    paths = [
        Path(package.locate_file(f"physx_usd_schemas/{module}/resources"))
        for module in ("PhysxSchema", "OmniUsdPhysicsDeformableSchema")
    ]
    for path in paths:
        for name in ("plugInfo.json", "generatedSchema.usda"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"Incomplete physx-usd-schemas: missing {path / name}. Reinstall the provider.")
    return paths
