# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared harness for the cable-rendering coverage.

Lives in the library rather than beside the tests because the cells are split across two modules
that cannot import one another: the kit-less renderers need a process where Kit was never started,
and Isaac RTX one where Kit started *before* anything imported USD. Both need the same scene,
framing and metrics, or the two halves stop being comparable.

A cable frozen at its spawn pose produces a perfectly stable, plausible pixel count, so presence
alone cannot gate this feature; and a cull leaving a few stray pixels still yields a centroid that
can drift far enough to satisfy a displacement check. Tracking is therefore asserted on centroid
displacement *and* retention, over a segmentation that refuses to measure a frame it cannot segment.
"""

from __future__ import annotations

import torch

# Fraction of the frame above which a "geometry" mask is not geometry but a mis-measurement. A cable
# occupies well under 1% of the frame; anything approaching totality means the mask has latched onto
# the backdrop, and every figure derived from it is meaningless.
MAX_PLAUSIBLE_LIT_FRACTION = 0.5
# A frozen cable still jitters by a pixel or two under resampling; real motion moves the centroid far
# further. Measured: frozen ~0.5-3 px over 140 steps, tracking >15 px.
MIN_CENTROID_SHIFT_PX = 8.0
# Measured mid-fall: 30 steps of free fall moves the cable well past the threshold above while
# keeping it inside the view, so a genuine miss cannot be confused with the cable leaving the frame.
TRACK_STEPS = 30
SPAWN_Z = 0.8
# Framed to keep the whole fall in view: the cable spawns near z=0.8 and settles at z~0.
EYE = (2.2, 2.2, 1.3)
TARGET = (0.0, 0.0, 0.35)
# OVRTX rejects camera prims outside the standard env namespace outright. Geometry must live there
# too: scene-partition primvars are authored by walking that subtree, so a prim outside it inherits
# no partition and the camera cannot see it.
ENV_NS = "/World/envs/env_0"


def look_at_quat(eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Camera orientation looking from ``eye`` to ``target``, as a world-convention ``(w, x, y, z)``.

    Baked into the camera cfg as an offset rather than applied afterwards with
    ``Camera.set_world_poses_from_view``, which does not stick: the camera stays at the origin and
    the cable falls out of frame, which is indistinguishable from the defect these cells catch.
    """
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    matrix = create_rotation_matrix_from_view(
        torch.tensor([eye], dtype=torch.float32),
        torch.tensor([target], dtype=torch.float32),
        "Z",
        device="cpu",
    )
    return tuple(float(value) for value in quat_from_matrix(matrix)[0].tolist())


def lit_mask(rgb, threshold: int = 12) -> torch.Tensor:
    """Boolean HxW mask of the pixels that differ from the frame's own background.

    Segments against the **modal** luminance, not an absolute threshold: a renderer with a bright
    backdrop saturates an absolute mask to the whole frame, pinning the centroid at the image
    centre, which reads exactly like a frozen render.
    """
    tensor = rgb.torch if hasattr(rgb, "torch") else rgb
    lum = tensor[..., :3].float().mean(dim=-1)
    while lum.dim() > 2:
        lum = lum[0]
    # The backdrop is whatever luminance dominates the frame. 64 bins is coarse enough to absorb
    # gradient and sampling noise into one bin, fine enough to leave the cable outside it.
    histogram = torch.histc(lum, bins=64, min=0.0, max=255.0)
    background = (int(torch.argmax(histogram).item()) + 0.5) * (255.0 / 64.0)
    return (lum - background).abs().gt(threshold)


def lit_pixel_count(rgb, threshold: int = 12) -> int:
    """How much cable is on screen. Companion to :func:`geometry_centroid`, which says *where*."""
    return int(lit_mask(rgb, threshold).sum().item())


def geometry_centroid(rgb, threshold: int = 12) -> tuple[float, float] | None:
    """Centroid of the lit pixels, or ``None`` when nothing is lit.

    Motion is asserted on the centroid rather than on a count of changed pixels: a frozen cable
    still re-rasterizes with sampling jitter, which registers thousands of "changed" pixels at a
    tiny per-pixel delta. The centroid does not move under jitter.
    """
    mask = lit_mask(rgb, threshold)
    if not bool(mask.any()):
        return None
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return float(ys.float().mean().item()), float(xs.float().mean().item())


def assert_mask_is_measuring_geometry(mask: torch.Tensor, renderer: str) -> None:
    """Fail loudly when the mask has latched onto the backdrop instead of the cable."""
    fraction = float(mask.float().mean().item())
    assert fraction < MAX_PLAUSIBLE_LIT_FRACTION, (
        f"{renderer}: {100 * fraction:.0f}% of the frame is marked as geometry, so the background"
        " segmentation has failed and no centroid or retention figure from this frame means"
        " anything. This is a broken measurement, not a renderer result."
    )


def cable_cfg(prim_path: str | None = None):
    """The cable under test: eleven control points, spawned high enough to fall a measurable way."""
    from isaaclab.assets import CableObjectCfg
    from isaaclab.sim import UsdPhysicsCollisionCfg
    from isaaclab.sim.spawners.materials import CableMaterialCfg
    from isaaclab.sim.spawners.shapes import CableCfg

    return CableObjectCfg(
        prim_path=prim_path or f"{ENV_NS}/Cable",
        spawn=CableCfg(
            positions=[(0.05 * index, 0.0, 0.0) for index in range(11)],
            physics_material=CableMaterialCfg(
                thickness=0.01,
                density=100.0,
                stretch_stiffness=3.18309886e8,
                bend_stiffness=2.03718327e9,
            ),
            collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SPAWN_Z)),
    )


def camera_cfg(renderer_cfg, prim_path: str | None = None, eye=EYE, target=TARGET):
    """A camera framed on the cable, with the pose baked into the cfg. See :func:`look_at_quat`."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path=prim_path or f"{ENV_NS}/Camera",
        width=320,
        height=240,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat(eye, target), convention="opengl"),
        renderer_cfg=renderer_cfg,
    )


def spawn_light() -> None:
    """A dome light, and deliberately no ground plane.

    A ground plane fills ~18k pixels of a 320x240 frame against a few hundred for the cable, so it
    dominates the mask and pins the centroid on a static silhouette. The cable need not land.
    """
    import isaaclab.sim as sim_utils

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)


def sim_cfg():
    """Physics for the cable cells. The device is deliberately left at the default.

    Rendering happens on the GPU, and forcing ``device="cpu"`` — as the physics-only cable tests do —
    crashes the OVRTX transform sync on a device mismatch.
    """
    from isaaclab_newton.physics import NewtonCfg

    import isaaclab.sim as sim_utils

    from isaaclab_contrib.deformable import VBDSolverCfg

    return sim_utils.SimulationCfg(
        dt=0.01,
        physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=20), num_substeps=8, use_cuda_graph=False),
    )


def assert_tracks(renderer, first_lit, first_centroid, last_lit, last_centroid, first_z, last_z, envs=1):
    """Assert the render followed the simulation, using both gates.

    The two are complementary, not redundant. A frozen render keeps its pixels (retention ~1.0, which
    the retention gate cannot see) and does not move (centroid ~0, which catches it). A culled render
    moves whatever remnant survives (centroid can pass) while losing most of its pixels (retention
    catches it). Dropping either lets one of the two real defects this suite exists to catch through.
    """
    import math

    where = f" across {envs} envs" if envs > 1 else ""
    assert first_z - last_z > 0.1, f"cable did not fall{where}: {first_z:.4f} -> {last_z:.4f}"
    assert first_centroid is not None, f"no cable geometry rendered in the first frame{where}"
    assert last_centroid is not None, f"{renderer} dropped the cable from the render as it moved{where}"
    assert last_lit >= 0.5 * first_lit, (
        f"{renderer} culled the cable while it moved{where}: {first_lit} -> {last_lit} lit px"
        f" ({100 * (1 - last_lit / max(first_lit, 1)):.0f}% lost). A surviving remnant still produces"
        " a centroid, so the displacement gate alone cannot detect this."
    )
    shift = math.dist(first_centroid, last_centroid)
    assert shift > MIN_CENTROID_SHIFT_PX, (
        f"{renderer} did not move the rendered cable{where} while it fell {first_z - last_z:.3f}"
        f" units: centroid {first_centroid} -> {last_centroid} ({shift:.1f} px, need >"
        f" {MIN_CENTROID_SHIFT_PX})"
    )
