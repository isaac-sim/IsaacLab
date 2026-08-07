# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact renderer/backend coverage matrix for the canonical rendering scene."""

from __future__ import annotations

from dataclasses import dataclass

SIMPLE_SHADING_AOVS = (
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
)
RTX_AOVS = (
    "rgb",
    "albedo",
    "semantic_segmentation",
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "normals",
    "instance_segmentation",
    "instance_id_segmentation_fast",
    "motion_vectors",
)
OVRTX_AOVS = tuple(aov for aov in RTX_AOVS if aov != "instance_id_segmentation_fast")
NEWTON_WARP_AOVS = (
    "rgb",
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "normals",
    "semantic_segmentation",
    "instance_segmentation",
)


@dataclass(frozen=True)
class RenderCase:
    """One scene construction that may validate several compatible AOVs."""

    physics: str
    renderer: str
    aovs: tuple[str, ...]
    profile: str = "standard"
    background_color: tuple[float, float, float] | None = None

    @property
    def id(self) -> str:
        profile = f"standard-{self.aovs[0]}" if self.profile == "standard" and len(self.aovs) == 1 else self.profile
        return f"{self.physics}-{self.renderer}-{profile}"

    def golden_id(self, aov: str) -> str:
        """Return a baseline identity that is stable when compatible AOVs are bundled."""
        suffix = aov if self.profile == aov and self.aovs == (aov,) else f"{self.profile}-{aov}"
        return f"{self.physics}-{self.renderer}-{suffix}"


def _single_aov_cases(physics: str, renderer: str, aovs: tuple[str, ...]) -> tuple[RenderCase, ...]:
    return tuple(RenderCase(physics, renderer, (aov,)) for aov in aovs)


def _simple_shading_cases(physics: str, renderer: str) -> tuple[RenderCase, ...]:
    # Simple shading modes share one renderer setting and cannot coexist in a camera.
    return tuple(RenderCase(physics, renderer, (aov,), profile=aov) for aov in SIMPLE_SHADING_AOVS)


_KIT_CASES = [
    RenderCase("physx", "isaac_rtx", RTX_AOVS),
    RenderCase("newton", "isaac_rtx", RTX_AOVS),
    RenderCase("physx", "newton_warp", NEWTON_WARP_AOVS),
    RenderCase("newton", "newton_warp", NEWTON_WARP_AOVS),
    *_simple_shading_cases("physx", "isaac_rtx"),
    RenderCase("physx", "isaac_rtx", ("rgb",), profile="yellow-background", background_color=(1.0, 1.0, 0.0)),
]
KIT_CASES = tuple(_KIT_CASES)

_KITLESS_CASES = [
    *_single_aov_cases("ovphysx", "ovrtx", OVRTX_AOVS),
    *_single_aov_cases("newton", "ovrtx", OVRTX_AOVS),
    RenderCase("ovphysx", "newton_warp", NEWTON_WARP_AOVS),
    RenderCase("newton", "newton_warp", NEWTON_WARP_AOVS),
    *_simple_shading_cases("ovphysx", "ovrtx"),
    RenderCase("ovphysx", "ovrtx", ("rgb",), profile="yellow-background", background_color=(1.0, 1.0, 0.0)),
]
KITLESS_CASES = tuple(
    (stage, case)
    for case in _KITLESS_CASES
    for stage in (("legacy", "ovstage") if case.renderer == "ovrtx" else ("legacy",))
)


def select_kitless_cases(stage: str, physics: str) -> tuple[tuple[str, RenderCase], ...]:
    """Select a bounded native-renderer lifecycle partition without duplicating case ownership."""
    return tuple(
        (case_stage, case) for case_stage, case in KITLESS_CASES if case_stage == stage and case.physics == physics
    )
