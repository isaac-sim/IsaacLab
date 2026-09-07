# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral kinematic rigid-object rendering contract.

Backend test modules provide only a simulation context and renderer configuration.
This module owns the scene, motion sequence, measurements, and assertions.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.renderers import RendererCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

__all__ = ["RigidObjectRenderingBackend", "run_rigid_object_scale_and_pose_rendering_contract"]

_NUM_ENVS = 2
_ENV_SPACING = 4.0
_CAMERA_DISTANCE = 2.0
_CAMERA_HEIGHT = 120
_CAMERA_WIDTH = 160
_OBJECT_SCALE = (1.0, 1.0, 8.0)
_OBJECT_SHIFT = 0.45
_MIN_OBJECT_DEPTH = 0.05
_MAX_OBJECT_DEPTH = 10.0
_MIN_OBJECT_PIXELS = 50
_MIN_CENTROID_SHIFT = 10.0


@dataclass(frozen=True)
class RigidObjectRenderingBackend:
    """Backend-owned inputs to the shared rendering contract."""

    name: str
    simulation_context_factory: Callable[[], AbstractContextManager[SimulationContext]]
    renderer_cfg: RendererCfg
    with_articulation: bool = False
    cleanup: Callable[[], None] | None = None


def _make_scene_cfg(backend: RigidObjectRenderingBackend) -> InteractiveSceneCfg:
    """Create the scene whose rendered behavior is shared by every backend."""

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        rigid_object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True)],
                scale=_OBJECT_SCALE,
            ),
        )
        camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            height=_CAMERA_HEIGHT,
            width=_CAMERA_WIDTH,
            update_period=0.0,
            update_latest_camera_pose=True,
            data_types=["depth"],
            renderer_cfg=backend.renderer_cfg,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(_MIN_OBJECT_DEPTH, 100.0),
            ),
        )
        if backend.with_articulation:
            articulation: ArticulationCfg = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Articulation",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd"
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -20.0, 0.0)),
                actuators={
                    "joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0),
                },
            )

    return _SceneCfg(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING, lazy_sensor_update=False)


def _require(condition: torch.Tensor | bool, message: str) -> None:
    """Raise a diagnostic assertion instead of relying on pytest rewriting this helper."""
    if not bool(condition):
        raise AssertionError(message)


def _measure_depth_mask(
    depth: torch.Tensor, backend_name: str, camera: Camera
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return silhouette height, width, and horizontal centroid for every camera."""
    depth_image = depth[..., 0]
    valid = torch.isfinite(depth_image) & (depth_image > _MIN_OBJECT_DEPTH) & (depth_image < _MAX_OBJECT_DEPTH)
    pixel_counts = valid.sum(dim=(1, 2))
    finite = torch.where(torch.isfinite(depth_image), depth_image, torch.zeros_like(depth_image))
    diagnostics = (
        f"depth min={finite.amin(dim=(1, 2)).tolist()} max={finite.amax(dim=(1, 2)).tolist()} "
        f"finite%={(torch.isfinite(depth_image).float().mean(dim=(1, 2)) * 100).tolist()} "
        f"camera pos_w={camera.data.pos_w.torch.tolist()}"
    )
    _require(
        torch.all(pixel_counts >= _MIN_OBJECT_PIXELS),
        f"[{backend_name}] Expected at least {_MIN_OBJECT_PIXELS} object pixels per camera, "
        f"got {pixel_counts.tolist()}; {diagnostics}.",
    )

    silhouette_heights = valid.any(dim=2).sum(dim=1)
    silhouette_widths = valid.any(dim=1).sum(dim=1)
    image_x = torch.arange(depth.shape[2], device=depth.device, dtype=torch.float32)
    centroids_x = (valid * image_x.view(1, 1, -1)).sum(dim=(1, 2)) / pixel_counts
    return silhouette_heights, silhouette_widths, centroids_x


def _write_pose_and_render(
    sim: SimulationContext,
    scene: InteractiveScene,
    rigid_object: RigidObject,
    camera: Camera,
    root_poses: torch.Tensor,
) -> torch.Tensor:
    """Write a kinematic pose, settle renderer state, and return depth."""
    rigid_object.write_root_pose_to_sim_index(root_pose=root_poses)
    for _ in range(3):
        sim.step()
        scene.update(sim.cfg.dt)
    torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch, root_poses, rtol=0.0, atol=1.0e-4)
    return camera.data.output["depth"].torch.clone()


def run_rigid_object_scale_and_pose_rendering_contract(backend: RigidObjectRenderingBackend) -> None:
    """Assert root-scale preservation and pose-to-pixel synchronization."""
    with backend.simulation_context_factory() as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_make_scene_cfg(backend))
        sim.register_interactive_scene(scene)
        rigid_object = scene["rigid_object"]
        camera = scene["camera"]

        try:
            sim.reset()
            scene.reset()
            _require(rigid_object.is_initialized, f"[{backend.name}] Rigid object did not initialize.")
            if backend.with_articulation:
                _require(scene["articulation"].is_initialized, f"[{backend.name}] Articulation did not initialize.")

            camera_eyes = scene.env_origins.clone()
            camera_eyes[:, 1] -= _CAMERA_DISTANCE
            camera.set_world_poses_from_view(camera_eyes, scene.env_origins)

            center_poses = torch.zeros((_NUM_ENVS, 7), device=rigid_object.device)
            center_poses[:, :3] = scene.env_origins
            center_poses[:, 6] = 1.0

            center_depth = _write_pose_and_render(sim, scene, rigid_object, camera, center_poses)
            center_heights, center_widths, center_centroids = _measure_depth_mask(center_depth, backend.name, camera)
            _require(
                torch.all(center_heights > 3 * center_widths),
                f"[{backend.name}] Expected root-scaled cubes to render as tall silhouettes, got "
                f"heights={center_heights.tolist()} and widths={center_widths.tolist()}.",
            )
            _require(
                torch.all(center_heights > _CAMERA_HEIGHT // 4),
                f"[{backend.name}] Expected root-scaled silhouettes to span more than one quarter of the image, "
                f"got heights={center_heights.tolist()}.",
            )
            _require(
                torch.all(torch.abs(center_centroids - (_CAMERA_WIDTH - 1) / 2) < 8.0),
                f"[{backend.name}] Expected centered silhouettes, got centroids={center_centroids.tolist()}.",
            )

            negative_poses = center_poses.clone()
            negative_poses[:, 0] -= _OBJECT_SHIFT
            negative_depth = _write_pose_and_render(sim, scene, rigid_object, camera, negative_poses)
            _, _, negative_centroids = _measure_depth_mask(negative_depth, backend.name, camera)

            positive_poses = center_poses.clone()
            positive_poses[:, 0] += _OBJECT_SHIFT
            positive_depth = _write_pose_and_render(sim, scene, rigid_object, camera, positive_poses)
            _, _, positive_centroids = _measure_depth_mask(positive_depth, backend.name, camera)

            negative_delta = negative_centroids - center_centroids
            positive_delta = positive_centroids - center_centroids
            _require(
                torch.all(negative_delta.abs() > _MIN_CENTROID_SHIFT),
                f"[{backend.name}] Negative-shift centroids moved too little: {negative_delta.tolist()}.",
            )
            _require(
                torch.all(positive_delta.abs() > _MIN_CENTROID_SHIFT),
                f"[{backend.name}] Positive-shift centroids moved too little: {positive_delta.tolist()}.",
            )
            _require(
                torch.all(negative_delta * positive_delta < 0.0),
                f"[{backend.name}] Opposite translations must move silhouettes in opposite directions, got "
                f"deltas {negative_delta.tolist()} and {positive_delta.tolist()}.",
            )
        finally:
            sim.register_interactive_scene(None)
            # Release camera-owned render products before another parametrized case creates a stage.
            camera._invalidate_initialize_callback(None)  # noqa: SLF001
            del camera, rigid_object, scene
            gc.collect()
            if backend.cleanup is not None:
                backend.cleanup()
