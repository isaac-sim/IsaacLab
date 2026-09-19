# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version compatibility between the public OVStage 0.1 API and OVStage 0.2 and later.

OVStage computes world transforms from the prim hierarchy either on the host or on the
device. The host model costs CPU work proportional to the number of prims, which the
barrier in the OVRTX ovstage write path then waits on, so the device model is preferred
wherever it is available. ``GPU_INCREMENTAL`` leaves objects out of place on OVStage 0.1
and is corrected in 0.2, so the model is chosen from the installed version rather than
hard-coded.

The installed version cannot change while the process runs, so the model is resolved once
at import and published as :data:`HIERARCHY_COMPUTATION_MODEL`.

The published name is resolved against :class:`ovstage.HierarchyComputationModel` by the
caller, which keeps this module free of an ``ovstage`` import and therefore importable
wherever the version policy needs to be inspected or tested.

The optional dependency is pinned to ``ovstage==0.2.0.377349``. Missing or invalid
version metadata selects ``CPU_INCREMENTAL``.
"""

from __future__ import annotations

import importlib.metadata
import logging

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# First OVStage version whose GPU_INCREMENTAL hierarchy model places objects correctly.
_GPU_HIERARCHY_VERSION = Version("0.2")


def detect_ovstage_version() -> Version | None:
    """Return the installed ``ovstage`` version.

    Read from distribution metadata rather than importing ``ovstage`` so the version
    policy can be inspected without loading the runtime. An unparsable version is logged
    and reported as missing, which keeps the OVStage 0.1 behavior.

    Returns:
        The installed version, or ``None`` when ``ovstage`` is absent or its version
        string cannot be parsed.
    """
    try:
        raw = importlib.metadata.version("ovstage")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return Version(raw)
    except InvalidVersion:
        logger.warning("Could not parse ovstage version %r; assuming the OVStage 0.1 hierarchy model.", raw)
        return None


def supports_gpu_hierarchy_computation(version: Version | None) -> bool:
    """Return whether ``version`` computes the prim hierarchy correctly on the device.

    Args:
        version: OVStage version to classify, or ``None`` when OVStage is unavailable.

    Returns:
        Whether ``version`` is OVStage 0.2 or newer.
    """
    return version is not None and version >= _GPU_HIERARCHY_VERSION


def resolve_hierarchy_computation_model(version: Version | None) -> str:
    """Return the :class:`ovstage.HierarchyComputationModel` member name for ``version``.

    Args:
        version: OVStage version the model is chosen for, or ``None`` when OVStage is
            unavailable.

    Returns:
        ``"GPU_INCREMENTAL"`` on OVStage 0.2 and later, otherwise ``"CPU_INCREMENTAL"``.
    """
    if supports_gpu_hierarchy_computation(version):
        return "GPU_INCREMENTAL"
    return "CPU_INCREMENTAL"


OVSTAGE_VERSION: Version | None = detect_ovstage_version()
"""Installed OVStage version, or ``None`` when it is unavailable or unparsable."""

HIERARCHY_COMPUTATION_MODEL: str = resolve_hierarchy_computation_model(OVSTAGE_VERSION)
"""Name of the hierarchy computation model to request for the installed OVStage."""
