# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable (``UsdGeom.BasisCurves``) rendering coverage for the kit-less renderers.

A cable's control points are rewritten every frame from Newton segment bodies. Nothing in the suite
rendered one before: the cable-object tests all run with ``render=False``, so a renderer could draw a
cable at its spawn pose and never move it again without any test noticing.

Two properties are asserted, in order, for every renderer:

  1. a cable renders at all, and
  2. the rendered image tracks the simulation instead of freezing at the spawn pose.

The second is the load-bearing one. See :mod:`isaaclab.test.utils.cable_rendering` for why the
metrics are shaped the way they are.

**Isaac RTX is covered separately**, in
``source/isaaclab_physx/test/renderers/test_cable_rendering_isaac_rtx.py``. It needs Kit started
before anything imports USD, which cannot be arranged in this module without making the kit-less
cells run under Kit — at which point they would no longer be testing the kit-less path.
"""

import pytest
import torch

pytest.importorskip("isaaclab_newton")
pytest.importorskip("newton")

from isaaclab_newton.assets import CableObject as NewtonCableObject

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera
from isaaclab.sim import build_simulation_context
from isaaclab.test.utils.cable_rendering import (
    ENV_NS,
    TRACK_STEPS,
    assert_mask_is_measuring_geometry,
    assert_tracks,
    cable_cfg,
    camera_cfg,
    geometry_centroid,
    lit_mask,
    lit_pixel_count,
    sim_cfg,
    spawn_light,
)
from isaaclab.utils.configclass import configclass

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cable rendering requires a GPU"),
]

# Minimum OVRTX that can draw an animated, partitioned cable at all. Below this the renderer
# publishes an empty bounding box for a partitioned BasisCurves and the whole partition goes dark, so
# a failure says nothing about Isaac Lab. A version floor rather than an xfail: an xfail records the
# cell as expected-to-fail and quietly stays red once the renderer is fixed, whereas a skip names the
# reason and turns into a real result the moment the floor is met.
_MIN_OVRTX = (0, 4, 1)

_RENDERERS = ["ovrtx", "newton_warp"]


def _skip_if_ovrtx_too_old() -> None:
    """Skip the OVRTX cells when the installed OVRTX predates the curve bounding-box fixes."""
    ovrtx = pytest.importorskip("ovrtx")
    raw = getattr(ovrtx, "__version__", "0.0.0")
    parts = []
    for piece in raw.split(".")[:3]:
        digits = "".join(character for character in piece if character.isdigit())
        parts.append(int(digits) if digits else 0)
    version = tuple(parts) + (0,) * (3 - len(parts))
    if version < _MIN_OVRTX:
        pytest.skip(
            f"OVRTX {raw} predates {'.'.join(str(value) for value in _MIN_OVRTX)}, the first build"
            " that renders a BasisCurves inside a scene partition at all. Cable rendering cannot be"
            " assessed below this floor."
        )


def _renderer_cfg(renderer: str):
    """Build the renderer config, skipping the case when that backend is not installed."""
    if renderer == "ovrtx":
        pytest.importorskip("isaaclab_ov")
        _skip_if_ovrtx_too_old()
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    if renderer == "newton_warp":
        from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg()
    raise ValueError(f"Unknown renderer: {renderer}")


def _build_scene(renderer: str) -> Camera:
    """Spawn the light and camera. Returns the camera."""
    spawn_light()
    return Camera(camera_cfg(_renderer_cfg(renderer)))


@pytest.mark.parametrize("renderer", _RENDERERS)
def test_cable_renders(renderer):
    """A cable must produce geometry in a render."""
    cfg = sim_cfg()
    with build_simulation_context(sim_cfg=cfg) as sim:
        camera = _build_scene(renderer)
        NewtonCableObject(cable_cfg())
        sim.reset()
        sim.step(render=False)
        camera.update(cfg.dt)

        mask = lit_mask(camera.data.output["rgb"])
        # Both bounds matter. Zero means nothing was drawn; near-totality means the background
        # segmentation failed, which would silently disarm the tracking cells that follow.
        assert_mask_is_measuring_geometry(mask, renderer)
        assert int(mask.sum().item()) > 0, f"{renderer} rendered no cable geometry"


@pytest.mark.parametrize("renderer", _RENDERERS)
def test_cable_render_tracks_simulation(renderer):
    """The rendered image must follow the cable as it falls, not freeze at the spawn pose."""
    cfg = sim_cfg()
    with build_simulation_context(sim_cfg=cfg) as sim:
        camera = _build_scene(renderer)
        cable = NewtonCableObject(cable_cfg())
        sim.reset()
        sim.step(render=False)
        cable.update(cfg.dt)
        camera.update(cfg.dt)
        # Compute eagerly: ``camera.data.output["rgb"]`` is a live buffer overwritten in place on the
        # next update, so holding the reference and measuring it later reads the LAST frame twice and
        # the comparison silently becomes a no-op.
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), renderer)
        first_lit = lit_pixel_count(camera.data.output["rgb"])
        first_centroid = geometry_centroid(camera.data.output["rgb"])
        first_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        for _ in range(TRACK_STEPS):
            sim.step(render=False)
            cable.update(cfg.dt)
        camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), renderer)
        last_lit = lit_pixel_count(camera.data.output["rgb"])
        last_centroid = geometry_centroid(camera.data.output["rgb"])
        last_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        assert_tracks(renderer, first_lit, first_centroid, last_lit, last_centroid, first_z, last_z)


@pytest.mark.parametrize("renderer", _RENDERERS)
def test_cable_renders_across_environments(renderer):
    """Cables must still render and track once the scene is replicated across environments.

    Replication is exercised elsewhere, but every test there steps with ``render=False``, so the
    *rendering* half of replication had no coverage at all.

    Two framing rules are load-bearing. The camera does **not** move with ``num_envs``: pulling it
    back to frame more environments shrinks each cable below the detection threshold, which reads
    exactly like a cull. And it is placed through the cfg, never with ``set_world_poses_from_view``.
    """
    num_envs = 4
    spacing = 0.6
    eye = (spacing, -1.6 * spacing, 1.15)
    target = (spacing, 0.0, 0.55)

    @configclass
    class _CableSceneCfg(InteractiveSceneCfg):
        cable = cable_cfg().replace(prim_path="{ENV_REGEX_NS}/Cable")
        camera = camera_cfg(_renderer_cfg(renderer), f"{ENV_NS}/MultiEnvCam", eye, target)

    cfg = sim_cfg()
    with build_simulation_context(sim_cfg=cfg) as sim:
        spawn_light()
        scene = InteractiveScene(_CableSceneCfg(num_envs=num_envs, env_spacing=spacing))
        sim.reset()
        cable = scene["cable"]
        camera = scene["camera"]

        assert cable.num_instances == num_envs, (
            f"replication produced {cable.num_instances} cables, expected {num_envs}"
        )

        scene.update(cfg.dt)
        camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), renderer)
        first_lit = lit_pixel_count(camera.data.output["rgb"])
        first_centroid = geometry_centroid(camera.data.output["rgb"])
        first_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        for _ in range(TRACK_STEPS):
            sim.step(render=False)
            scene.update(cfg.dt)
        camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), renderer)
        last_lit = lit_pixel_count(camera.data.output["rgb"])
        last_centroid = geometry_centroid(camera.data.output["rgb"])
        last_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        assert_tracks(renderer, first_lit, first_centroid, last_lit, last_centroid, first_z, last_z, envs=num_envs)
