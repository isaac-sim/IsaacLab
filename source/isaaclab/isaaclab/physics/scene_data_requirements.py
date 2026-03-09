# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central requirement resolution for scene-data consumers.

This module is intentionally type-based (not config-import based) so requirement
checks stay robust even when optional backend packages are not installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneDataRequirement:
    """Capabilities required from a scene data provider."""

    requires_newton_model: bool = False
    requires_usd_stage: bool = False


_VISUALIZER_REQUIREMENTS: dict[str, SceneDataRequirement] = {
    "kit": SceneDataRequirement(requires_usd_stage=True),
    "newton": SceneDataRequirement(requires_newton_model=True),
    "rerun": SceneDataRequirement(requires_newton_model=True),
    "viser": SceneDataRequirement(requires_newton_model=True),
}

_RENDERER_REQUIREMENTS: dict[str, SceneDataRequirement] = {
    "isaac_rtx": SceneDataRequirement(requires_usd_stage=True),
    "newton_warp": SceneDataRequirement(requires_newton_model=True),
    "ovrtx": SceneDataRequirement(requires_newton_model=True, requires_usd_stage=True),
}


def supported_visualizer_types() -> tuple[str, ...]:
    """Return supported visualizer type names in sorted order."""
    return tuple(sorted(_VISUALIZER_REQUIREMENTS))


def supported_renderer_types() -> tuple[str, ...]:
    """Return supported renderer type names in sorted order."""
    return tuple(sorted(_RENDERER_REQUIREMENTS))


def requirement_for_visualizer_type(visualizer_type: str) -> SceneDataRequirement:
    """Resolve scene-data requirements for one visualizer type.

    Raises:
        ValueError: If ``visualizer_type`` is unknown.
    """
    requirement = _VISUALIZER_REQUIREMENTS.get(visualizer_type)
    if requirement is None:
        supported = ", ".join(repr(v) for v in supported_visualizer_types())
        raise ValueError(f"Unknown visualizer type {visualizer_type!r}. Supported types: {supported}.")
    return requirement


def requirement_for_renderer_type(renderer_type: str) -> SceneDataRequirement:
    """Resolve scene-data requirements for one renderer type.

    Raises:
        ValueError: If ``renderer_type`` is unknown.
    """
    requirement = _RENDERER_REQUIREMENTS.get(renderer_type)
    if requirement is None:
        supported = ", ".join(repr(v) for v in supported_renderer_types())
        raise ValueError(f"Unknown renderer type {renderer_type!r}. Supported types: {supported}.")
    return requirement


def aggregate_requirements(requirements: Iterable[SceneDataRequirement]) -> SceneDataRequirement:
    """Combine a sequence of requirements using logical OR."""
    requires_newton_model = False
    requires_usd_stage = False
    for requirement in requirements:
        requires_newton_model |= requirement.requires_newton_model
        requires_usd_stage |= requirement.requires_usd_stage
    return SceneDataRequirement(requires_newton_model=requires_newton_model, requires_usd_stage=requires_usd_stage)
