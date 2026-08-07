# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable (``UsdGeom.BasisCurves``) rendering coverage for the Isaac RTX renderer.

The kit-less half of this coverage lives in
``source/isaaclab/test/renderers/test_cable_rendering.py``; the shared scene, framing and metrics
live in :mod:`isaaclab.test.utils.cable_rendering` so the two halves stay comparable.

**Why this is a separate module.** Isaac RTX needs Kit started *before* anything imports USD, hence
the ``AppLauncher`` call at module scope below. Booting Kit lazily arrives too late — the
module-scope imports have already pulled in USD, and USD-first dies in ``libusd_tf``. Booting it for
all renderers would stop the OVRTX and Newton Warp cells exercising the kit-less path they exist to
cover. Splitting the module gives each half the process it needs.
"""

from isaaclab.app import AppLauncher

# Must precede every USD-touching import below. ``enable_cameras`` selects the rendering Kit
# experience; without it no offscreen render product is created and the annotator returns nothing,
# which reads as a renderer that draws no cable.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch
import warp as wp
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

# Newton's VBD solver, which cables require, needs symbols that landed in Warp 1.15
# (``DeterministicMode``) and 1.16 (``quat_twist_angle_signed``). Below that the cells die deep in
# Newton's lazy solver shim as ``ImportError: cannot import name 'SolverVBD'``, which names the wrong
# thing entirely.
_MIN_WARP = (1, 16)


def _warp_version() -> tuple[int, int]:
    """Major/minor of the Warp this process actually imported."""
    parts = []
    for piece in wp.__version__.split(".")[:2]:
        digits = "".join(character for character in piece if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) + (0,) * (2 - len(parts))


# Keyed on the IMPORTED module, not a listing of Isaac Sim's ``extscache``. Isaac Sim bundles its
# own Warp, but whether that copy reaches ``sys.path`` ahead of the environment's depends on the
# venv layout, so a gate that globs ``extscache`` both skips runs that would pass and passes runs
# that would skip. Evaluated after ``AppLauncher``, so it reflects what Kit actually loaded.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cable rendering requires a GPU"),
    pytest.mark.skipif(
        _warp_version() < _MIN_WARP,
        reason=(
            f"the imported Warp is {wp.__version__}, but Newton's VBD solver (required by cables)"
            f" needs >= {_MIN_WARP[0]}.{_MIN_WARP[1]}. Isaac Sim bundles its own Warp under"
            " extscache; if that copy is winning on sys.path, replacing its payload with a newer"
            " Warp clears this — nothing pins the version."
        ),
    ),
]

_RENDERER = "isaac_rtx"


def _renderer_cfg():
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    return IsaacRtxRendererCfg()


def _build_scene() -> Camera:
    spawn_light()
    return Camera(camera_cfg(_renderer_cfg()))


def test_cable_renders():
    """A cable must produce geometry in a render."""
    cfg = sim_cfg()
    with build_simulation_context(sim_cfg=cfg) as sim:
        camera = _build_scene()
        NewtonCableObject(cable_cfg())
        sim.reset()
        sim.step(render=False)
        camera.update(cfg.dt)

        mask = lit_mask(camera.data.output["rgb"])
        assert_mask_is_measuring_geometry(mask, _RENDERER)
        assert int(mask.sum().item()) > 0, f"{_RENDERER} rendered no cable geometry"


def test_cable_render_tracks_simulation():
    """The rendered image must follow the cable as it falls, not freeze at the spawn pose."""
    cfg = sim_cfg()
    with build_simulation_context(sim_cfg=cfg) as sim:
        camera = _build_scene()
        cable = NewtonCableObject(cable_cfg())
        sim.reset()
        sim.step(render=False)
        cable.update(cfg.dt)
        camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), _RENDERER)
        first_lit = lit_pixel_count(camera.data.output["rgb"])
        first_centroid = geometry_centroid(camera.data.output["rgb"])
        first_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        for _ in range(TRACK_STEPS):
            sim.step(render=False)
            cable.update(cfg.dt)
        camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), _RENDERER)
        last_lit = lit_pixel_count(camera.data.output["rgb"])
        last_centroid = geometry_centroid(camera.data.output["rgb"])
        last_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        assert_tracks(_RENDERER, first_lit, first_centroid, last_lit, last_centroid, first_z, last_z)


def test_cable_renders_across_environments():
    """Cables must still render and track once the scene is replicated across environments."""
    num_envs = 4
    spacing = 0.6
    eye = (spacing, -1.6 * spacing, 1.15)
    target = (spacing, 0.0, 0.55)

    @configclass
    class _CableSceneCfg(InteractiveSceneCfg):
        cable = cable_cfg().replace(prim_path="{ENV_REGEX_NS}/Cable")
        camera = camera_cfg(_renderer_cfg(), f"{ENV_NS}/MultiEnvCam", eye, target)

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
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), _RENDERER)
        first_lit = lit_pixel_count(camera.data.output["rgb"])
        first_centroid = geometry_centroid(camera.data.output["rgb"])
        first_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        for _ in range(TRACK_STEPS):
            sim.step(render=False)
            scene.update(cfg.dt)
        camera.update(cfg.dt)
        assert_mask_is_measuring_geometry(lit_mask(camera.data.output["rgb"]), _RENDERER)
        last_lit = lit_pixel_count(camera.data.output["rgb"])
        last_centroid = geometry_centroid(camera.data.output["rgb"])
        last_z = cable.data.segment_pose_w.torch[..., 2].mean().item()

        assert_tracks(_RENDERER, first_lit, first_centroid, last_lit, last_centroid, first_z, last_z, envs=num_envs)
