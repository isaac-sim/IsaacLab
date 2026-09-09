# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera pose writes must take effect on every physics backend.

:meth:`Camera.set_world_poses` writes through the sensor's :class:`FrameView`. Under PhysX that view is
Fabric-backed (:class:`FabricFrameView`), so the RTX renderer -- which reads the USD/Fabric camera prim --
follows the write. Under Newton the view is a :class:`NewtonSiteFrameView`, which updates only in-memory
Warp state; nothing mirrors the pose back to the camera prim, so the rendered image keeps the old pose
even though ``camera.data.pos_w`` reports the new one.

Both tests move a downward-looking camera straight up over a ground plane, which multiplies the distance
to every visible surface, and check the two observable consequences of the write:

- ``_moves_reported_pose_*`` reads ``camera.data.pos_w`` (deterministic, no renderer involved).
- ``_moves_render_*`` compares the rendered depth before and after the move.

Set ``ISAACLAB_TEST_SAVE_IMAGES=1`` to dump the compared depth and RGB frames as PNGs under
``<this directory>/output/<test name>/``, which shows the Newton render sitting at the old pose.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import os

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

BACKEND_CFGS = [PhysxCfg(), NewtonCfg(solver_cfg=MJWarpSolverCfg())]
BACKEND_IDS = ["physx", "newton"]

# Dump the compared depth and RGB frames for inspection; off by default so the test writes nothing.
SAVE_IMAGES = os.environ.get("ISAACLAB_TEST_SAVE_IMAGES", "0") == "1"
IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@configclass
class _SceneCfg(InteractiveSceneCfg):
    """A single rigid body under the camera; Newton cannot build a model from an empty scene."""

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )


# Visual-only markers giving the render something to move against: a 0.5 m cube directly under the camera,
# which shrinks as the camera rises, flanked by two that sit outside the frame at the close height and only
# come into view at the far one. Coloured to stand out against the dark ground plane.
#
# They are spawned at ``/World`` rather than as scene entities because geometry placed under
# ``/World/envs/env_*`` by :class:`InteractiveScene` does not reach the RTX renderer in a bare
# :func:`build_simulation_context` (the same prim spawned at ``/World`` renders), so the scene cube above
# is invisible and serves only to give Newton a body to build its model from.
_MARKERS = (
    ((0.0, 0.0, 0.25), (0.9, 0.3, 0.1)),
    ((-1.5, 0.0, 0.25), (0.2, 0.6, 0.9)),
    ((1.5, 0.0, 0.25), (0.3, 0.8, 0.3)),
)


def _spawn_visual_markers() -> None:
    """Spawn the marker cubes as plain USD gprims, with no physics attached."""
    from pxr import Gf, UsdGeom

    for index, (translation, color) in enumerate(_MARKERS):
        # A UsdGeom.Cube has size 2.0, so a 0.25 scale gives a 0.5 m box.
        prim = sim_utils.create_prim(f"/World/Marker_{index}", "Cube", translation=translation, scale=(0.25,) * 3)
        geom = UsdGeom.Cube(prim)
        geom.CreateDisplayColorAttr()
        geom.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


def _save_images(
    depths: list[torch.Tensor], rgbs: list[torch.Tensor], heights_m: tuple[float, ...], output_subdir: str
) -> None:
    """Write a depth and an RGB PNG per camera height into ``IMAGE_OUTPUT_DIR/<output_subdir>/``.

    The two views show the move differently: depth reads it as overall brightness, RGB as more of the
    ground plane's grid falling inside the frame. Depth grey level maps against a fixed scale set by the
    highest commanded camera pose, rather than against each frame's own range: a camera that did not move
    then renders identically across heights, and the frames stay comparable across backends. Headroom
    covers the oblique corner rays, which are longer than the camera height.
    """
    from PIL import Image

    output_dir = os.path.join(IMAGE_OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    scale_m = 1.5 * max(heights_m)
    for height, depth, rgb in zip(heights_m, depths, rgbs):
        images = {
            "depth": (depth.squeeze(0).squeeze(-1) / scale_m * 255.0).clamp(0.0, 255.0).to(torch.uint8),
            "rgb": rgb.squeeze(0)[..., :3].to(torch.uint8),
        }
        for data_type, image in images.items():
            output_path = os.path.join(output_dir, f"{data_type}-{height:g}m.png")
            Image.fromarray(image.numpy()).save(output_path)
            print(f"Wrote {output_path}", flush=True)


def _capture_at_heights(
    physics_cfg, heights_m: tuple[float, ...], image_subdir: str | None = None
) -> list[tuple[torch.Tensor, float]]:
    """Move the camera to each height in turn, returning its ``(pos_w, mean visible depth [m])`` at each.

    The camera looks straight down at the ground plane, so the depth it reports is dominated by its height.
    It spawns at the first height rather than at the origin, so that a dropped pose write leaves a valid
    (but unchanged) depth image rather than an empty one.

    Args:
        physics_cfg: The physics backend configuration to build the simulation with.
        heights_m: Camera heights [m] above the ground plane to capture at, in order.
        image_subdir: When given and ``SAVE_IMAGES`` is set, the depth and RGB frames are written to
            this subdirectory of :obj:`IMAGE_OUTPUT_DIR`.
    """
    device = "cuda:0"
    # Physics steps taken after each pose write so the renderer produces a frame at the new pose.
    steps_per_pose = 2
    max_range_m = 1.0e5
    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        height=128,
        width=256,
        update_period=0,
        update_latest_camera_pose=True,
        data_types=["distance_to_camera", "rgb"],
        # OpenGL convention: the camera looks along its -Z axis, so identity orientation looks straight down.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, heights_m[0]), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, max_range_m)
        ),
    )

    sim_cfg = SimulationCfg(physics=physics_cfg, device=device)
    captures = []
    depths = []
    rgbs = []
    with build_simulation_context(device=device, sim_cfg=sim_cfg, add_ground_plane=True, add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.0))
        _spawn_visual_markers()
        camera = Camera(camera_cfg)
        sim.reset()

        orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        for height in heights_m:
            positions = torch.tensor([[0.0, 0.0, height]], device=device)
            camera.set_world_poses(positions, orientations, convention="opengl")
            for _ in range(steps_per_pose):
                sim.step()
            camera.update(sim.get_physics_dt())
            depth = camera.data.output["distance_to_camera"].torch.detach().float().cpu()
            # The sky renders at the far clipping range; average only over pixels that hit geometry.
            visible = depth[torch.isfinite(depth) & (depth < max_range_m)]
            assert visible.numel() > 0, "No valid depth pixels; the camera sees no geometry."
            captures.append((camera.data.pos_w.torch.detach().float().cpu().clone(), visible.mean().item()))
            # Sky pixels come back at the far clipping range; clamp them so they do not dominate the scale.
            depths.append(torch.nan_to_num(depth, posinf=0.0).clamp(max=visible.max().item()))
            rgbs.append(camera.data.output["rgb"].torch.detach().float().cpu())

        del camera

    if SAVE_IMAGES and image_subdir is not None:
        _save_images(depths, rgbs, heights_m, image_subdir)
    return captures


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_reported_pose(physics_cfg, request):
    """``camera.data.pos_w`` follows a ``set_world_poses`` write on every backend."""
    # Camera heights [m] above the ground plane, and the reported shift below which the write was dropped.
    close_m, far_m = 2.0, 8.0
    pose_shift_threshold_m = 0.5 * (far_m - close_m)

    (pos_close, _), (pos_far, _) = _capture_at_heights(physics_cfg, (close_m, far_m), request.node.name)

    np.testing.assert_allclose(pos_close.numpy(), [[0.0, 0.0, close_m]], atol=1e-3)
    np.testing.assert_allclose(pos_far.numpy(), [[0.0, 0.0, far_m]], atol=1e-3)
    shift = (pos_far - pos_close).norm(dim=-1).max().item()
    assert shift > pose_shift_threshold_m, (
        f"Expected camera.data.pos_w to follow the pose write (> {pose_shift_threshold_m} m); got {shift:.4f} m."
    )


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_render(physics_cfg, request):
    """The rendered depth follows a ``set_world_poses`` write on every backend.

    Under Newton the ``NewtonSiteFrameView`` write never reaches the camera prim the RTX renderer reads,
    so the image stays at the old pose and the depth ratio collapses to ~1.
    """
    # Camera heights [m] above the ground plane. The true far-to-close depth ratio is ~4x; 1.5 leaves room
    # for the ground plane filling different fractions of the frame while still failing hard if the render
    # does not move at all.
    close_m, far_m = 2.0, 8.0
    depth_ratio_threshold = 1.5

    (_, depth_close_m), (_, depth_far_m) = _capture_at_heights(physics_cfg, (close_m, far_m), request.node.name)

    ratio = depth_far_m / depth_close_m
    assert ratio > depth_ratio_threshold, (
        f"Far depth ({depth_far_m:.2f} m) should be > {depth_ratio_threshold}x close depth"
        f" ({depth_close_m:.2f} m); got {ratio:.2f}x. The camera pose write is not reaching the renderer."
    )
