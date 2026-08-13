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

import os

import pytest
import torch

pytest.importorskip("isaaclab_newton")
pytest.importorskip("newton")

from isaaclab_newton.assets import CableObject as NewtonCableObject

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera
from isaaclab.sim import build_simulation_context
from isaaclab.test.utils.cable_rendering import (
    ENV_SPACING,
    TRACK_STEPS,
    assert_mask_is_measuring_geometry,
    assert_tracks,
    cable_cfg,
    camera_cfg,
    capture_rgb_frame,
    geometry_centroid,
    lit_mask,
    lit_pixel_count,
    maybe_save_rgb_gif,
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

    Covers the *rendering* half of multi-env replication. Other clone tests step with
    ``render=False``, so they never check that replicated cable USD/render bindings still draw
    and follow motion.

    Uses the supported tiled-camera pattern: one camera prim per env via ``{ENV_REGEX_NS}``,
    each framed like the single-env cells so cables stay large enough for the lit-mask metrics.
    """
    num_envs = 4

    @configclass
    class _CableSceneCfg(InteractiveSceneCfg):
        # Clone cable and camera into every env (supported multi-env / tiled Camera contract).
        cable = cable_cfg().replace(prim_path="{ENV_REGEX_NS}/Cable")
        camera = camera_cfg(_renderer_cfg(renderer), "{ENV_REGEX_NS}/Camera")

    cfg = sim_cfg()
    with build_simulation_context(sim_cfg=cfg) as sim:
        # Dome light only (no ground plane) so lit-mask segments cables, not a big floor.
        spawn_light()
        scene = InteractiveScene(_CableSceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING))
        sim.reset()
        cable = scene["cable"]
        camera = scene["camera"]

        # Sanity: physics/cloner actually produced one cable instance per env.
        assert cable.num_instances == num_envs, (
            f"replication produced {cable.num_instances} cables, expected {num_envs}"
        )
        assert camera.num_instances == num_envs, (
            f"tiled camera has {camera.num_instances} views, expected {num_envs}"
        )

        # Baseline frame after reset: sync assets, render once, measure lit pixels / centroid / z.
        scene.update(cfg.dt)
        camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), renderer)
        first_lit = lit_pixel_count(camera.data.output["rgb"])
        first_centroid = geometry_centroid(camera.data.output["rgb"])
        first_z = cable.data.segment_pose_w.torch[..., 2].mean().item()
        recording = bool(os.environ.get("ISAAC_LAB_SAVE_CABLE_RENDERING_GIF"))
        frames = [capture_rgb_frame(camera.data.output["rgb"])] if recording else []

        # Keep normal runs cheap; opt-in GIF capture renders and records every physics step.
        for _ in range(TRACK_STEPS):
            sim.step(render=recording)
            scene.update(cfg.dt)
            if recording:
                camera.update(cfg.dt)
                frames.append(capture_rgb_frame(camera.data.output["rgb"]))
        # One render after motion; compare against the baseline.
        if not recording:
            camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), renderer)
        last_lit = lit_pixel_count(camera.data.output["rgb"])
        last_centroid = geometry_centroid(camera.data.output["rgb"])
        last_z = cable.data.segment_pose_w.torch[..., 2].mean().item()
        maybe_save_rgb_gif(frames, f"cable-rendering-{renderer}-multi-env.gif")

        # Gates: cables fell in sim, image centroid moved, and enough lit pixels remained
        # (catches frozen spawn-pose draws and post-motion culls).
        assert_tracks(renderer, first_lit, first_centroid, last_lit, last_centroid, first_z, last_z, envs=num_envs)
