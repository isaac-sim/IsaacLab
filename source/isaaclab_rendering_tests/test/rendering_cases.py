# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact renderer/backend coverage matrix for all golden rendering scenes."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.renderers.output_contract import RenderBufferKind

SIMPLE_SHADING_AOVS = (
    RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE,
    RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL,
    RenderBufferKind.SIMPLE_SHADING_FULL_MDL,
)
RTX_AOVS = (
    RenderBufferKind.RGB,
    RenderBufferKind.ALBEDO,
    RenderBufferKind.SEMANTIC_SEGMENTATION,
    RenderBufferKind.DEPTH,
    RenderBufferKind.DISTANCE_TO_CAMERA,
    RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
    RenderBufferKind.NORMALS,
    RenderBufferKind.INSTANCE_SEGMENTATION,
    RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST,
    RenderBufferKind.MOTION_VECTORS,
)
OVRTX_AOVS = tuple(aov for aov in RTX_AOVS if aov != RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST)
NEWTON_WARP_AOVS = (
    RenderBufferKind.RGB,
    RenderBufferKind.DEPTH,
    RenderBufferKind.DISTANCE_TO_CAMERA,
    RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
    RenderBufferKind.NORMALS,
    RenderBufferKind.SEMANTIC_SEGMENTATION,
    RenderBufferKind.INSTANCE_SEGMENTATION,
)
# Motion-vector magnitudes vary substantially across GPUs; the runner validates their behavior after one step.
NON_GOLDEN_AOVS = frozenset({RenderBufferKind.MOTION_VECTORS})


@dataclass(frozen=True)
class RenderCase:
    """One scene construction that may validate several compatible AOVs."""

    physics: str
    renderer: str
    aovs: tuple[RenderBufferKind, ...]
    variant: str | None = None
    scene: str = "rendering_scene"
    background_color: tuple[float, float, float] | None = None

    @property
    def id(self) -> str:
        variant = self.variant or (self.aovs[0].value if len(self.aovs) == 1 else None)
        return "-".join(part for part in (self.scene, self.physics, self.renderer, variant) if part)

    def golden_id(self, aov: RenderBufferKind) -> str:
        """Return a baseline identity that is stable when compatible AOVs are bundled."""
        suffix = aov.value if self.variant is None else f"{self.variant}-{aov.value}"
        return f"{self.physics}-{self.renderer}-{suffix}"

    def golden_filename(self, aov: RenderBufferKind, namespace: str | None = None) -> str:
        """Return a baseline filename independent of execution-path details."""
        prefix = f"{namespace}-" if namespace else ""
        return f"{prefix}{self.golden_id(aov)}.png"


def _single_aov_cases(physics: str, renderer: str, aovs: tuple[RenderBufferKind, ...]) -> tuple[RenderCase, ...]:
    return tuple(RenderCase(physics, renderer, (aov,)) for aov in aovs)


def _simple_shading_cases(physics: str, renderer: str) -> tuple[RenderCase, ...]:
    # Simple shading modes share one renderer setting and cannot coexist in a camera.
    return tuple(RenderCase(physics, renderer, (aov,)) for aov in SIMPLE_SHADING_AOVS)


_KIT_CASES = [
    RenderCase("physx", "isaac_rtx", RTX_AOVS),
    RenderCase("newton", "isaac_rtx", RTX_AOVS),
    RenderCase("physx", "newton_warp", NEWTON_WARP_AOVS),
    RenderCase("newton", "newton_warp", NEWTON_WARP_AOVS),
    *_simple_shading_cases("physx", "isaac_rtx"),
    RenderCase(
        "physx",
        "isaac_rtx",
        (RenderBufferKind.RGB,),
        variant="yellow-background",
        background_color=(1.0, 1.0, 0.0),
    ),
]
KIT_CASES = tuple(_KIT_CASES)

_KITLESS_CASES = [
    *_single_aov_cases("ovphysx", "ovrtx", OVRTX_AOVS),
    *_single_aov_cases("newton", "ovrtx", OVRTX_AOVS),
    RenderCase("ovphysx", "newton_warp", NEWTON_WARP_AOVS),
    RenderCase("newton", "newton_warp", NEWTON_WARP_AOVS),
    *_simple_shading_cases("ovphysx", "ovrtx"),
    RenderCase(
        "ovphysx",
        "ovrtx",
        (RenderBufferKind.RGB,),
        variant="yellow-background",
        background_color=(1.0, 1.0, 0.0),
    ),
]
# Ovstage selects OVRTX's USD scene-reader path, independently of whether Newton or OVPhysX advances physics.
KITLESS_CASES = tuple(
    (stage, case)
    for case in _KITLESS_CASES
    for stage in (("legacy", "ovstage") if case.renderer == "ovrtx" else ("legacy",))
)


@dataclass(frozen=True)
class SceneCoverage:
    aovs: tuple[RenderBufferKind, ...]
    kit_physics: tuple[str, ...]
    kitless_physics: tuple[str, ...]


SPECIALIZED_SCENE_COVERAGE = {
    "shadow_hand": SceneCoverage(
        (RenderBufferKind.RGB, RenderBufferKind.SEMANTIC_SEGMENTATION, RenderBufferKind.INSTANCE_SEGMENTATION),
        ("physx", "newton"),
        ("ovphysx", "newton"),
    ),
    "kuka_heterogeneous": SceneCoverage((RenderBufferKind.RGB, RenderBufferKind.INSTANCE_SEGMENTATION), ("physx",), ()),
    "franka_cloth": SceneCoverage((RenderBufferKind.RGB, RenderBufferKind.MOTION_VECTORS), ("newton",), ("newton",)),
    "franka_soft": SceneCoverage((RenderBufferKind.RGB, RenderBufferKind.MOTION_VECTORS), ("newton",), ("newton",)),
}
SPECIALIZED_KIT_CASES = tuple(
    RenderCase(
        physics,
        renderer,
        tuple(aov for aov in coverage.aovs if renderer != "newton_warp" or aov in NEWTON_WARP_AOVS),
        scene=scene,
    )
    for scene, coverage in SPECIALIZED_SCENE_COVERAGE.items()
    for physics in coverage.kit_physics
    for renderer in ("isaac_rtx", "newton_warp")
)
SPECIALIZED_KITLESS_CASES = tuple(
    (stage, case)
    for scene, coverage in SPECIALIZED_SCENE_COVERAGE.items()
    for physics in coverage.kitless_physics
    for case in (
        *(RenderCase(physics, "ovrtx", (aov,), scene=scene) for aov in coverage.aovs),
        RenderCase(
            physics,
            "newton_warp",
            tuple(
                aov
                for aov in coverage.aovs
                if aov in NEWTON_WARP_AOVS
                # Legacy USD cloning exposes either env_0 or its clones in semantic color depending on GPU.
                # The shared legacy and specialized Kit cases retain stable Newton-Warp semantic coverage.
                and not (scene == "shadow_hand" and aov == RenderBufferKind.SEMANTIC_SEGMENTATION)
            ),
            scene=scene,
        ),
    )
    for stage in (("legacy", "ovstage") if case.renderer == "ovrtx" and physics == "ovphysx" else ("legacy",))
)
KIT_RENDERING_CASES = KIT_CASES + SPECIALIZED_KIT_CASES
KITLESS_RENDERING_CASES = KITLESS_CASES + SPECIALIZED_KITLESS_CASES
