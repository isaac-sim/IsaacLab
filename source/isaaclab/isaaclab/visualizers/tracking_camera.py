# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene cameras that follow an asset, shown in the visualizer streaming view."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from .visualizer_cfg import USD_DEFAULT_VERTICAL_APERTURE_MM, TrackingCameraCfg, VisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..scene import InteractiveScene
    from ..sensors import Camera, CameraCfg


def tracking_camera_cfgs(sim_cfg) -> dict[str, tuple[TrackingCameraCfg, VisualizerCfg]]:
    """Return the tracking cameras the visualizers use with the first visualizer using each, keyed by sensor name.

    A visualizer with its own ``cameras`` uses those; otherwise it uses the ``cameras`` of
    :attr:`~isaaclab.sim.SimulationCfg.default_visualizer_cfg`.

    Raises:
        ValueError: If two visualizers declare different cameras under the same name.
    """
    visualizer_cfgs = sim_cfg.visualizer_cfgs
    visualizer_cfgs = visualizer_cfgs if isinstance(visualizer_cfgs, (list, tuple)) else [visualizer_cfgs]
    default_cfg = sim_cfg.default_visualizer_cfg
    cameras = {}
    for viewer in visualizer_cfgs or [default_cfg]:
        declaring = viewer if viewer.cameras is not None else default_cfg
        for camera in (declaring.cameras or []) if declaring is not None else []:
            if not isinstance(camera, TrackingCameraCfg):
                continue
            name = camera.prim_path.rsplit("/", 1)[-1]
            if name in cameras and cameras[name][0] != camera:
                raise ValueError(
                    f"Visualizers declare different tracking cameras named {name!r}; give each its own prim_path."
                )
            cameras.setdefault(name, (camera, viewer))
    return cameras


def make_scene_camera_cfg(cfg: TrackingCameraCfg, renderer_cfg, background_color=None) -> CameraCfg:
    """Return the scene :class:`~isaaclab.sensors.CameraCfg` that realizes *cfg* in every environment.

    The camera starts at the configured offsets, so a camera that tracks nothing stays fixed relative to its
    environment origin. Tracking cameras get their pose from :class:`TrackingCameraUpdater` instead.
    """
    from ..sensors import CameraCfg
    from ..sim import PinholeCameraCfg
    from ..utils.math import create_rotation_matrix_from_view, quat_from_matrix

    rotation = quat_from_matrix(create_rotation_matrix_from_view(torch.tensor([cfg.eye]), torch.tensor([cfg.lookat])))
    width, height = cfg.resolution
    return CameraCfg(
        prim_path=cfg.prim_path,
        width=width,
        height=height,
        data_types=list(cfg.data_types),
        # the viewport's vertical field of view for this focal length, so the camera frames the scene as it does
        spawn=PinholeCameraCfg(
            focal_length=cfg.focal_length,
            horizontal_aperture=USD_DEFAULT_VERTICAL_APERTURE_MM * width / height,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=cfg.eye, rot=tuple(rotation[0].tolist()), convention="opengl"),
        renderer_cfg=renderer_cfg,
        background_color=background_color,
    )


def add_tracking_cameras(env_cfg, sim_cfg, physics_cfg) -> bool:
    """Add the scene cameras that the launcher's visualizers declare to *env_cfg*'s scene.

    Visualizers exist only when ``--visualizer`` selects one or a video records from one, so runs that display
    nothing pay nothing. A camera without ``renderer_cfg`` uses the Newton Warp renderer on Newton physics and
    the camera default otherwise, and takes its background color from its visualizer.

    Args:
        env_cfg: The launched config; only one with a ``scene`` is changed.
        sim_cfg: Its :class:`~isaaclab.sim.SimulationCfg`, whose ``visualizer_cfgs`` the launcher has resolved.
        physics_cfg: The resolved physics config.

    Returns:
        Whether a camera was added, which the launcher's scan must then see.
    """
    from ..sensors import CameraCfg

    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is None or not sim_cfg.visualizer_cfgs:
        return False
    renderer_cfg = None
    if type(physics_cfg).__module__.startswith("isaaclab_newton"):
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        renderer_cfg = NewtonWarpRendererCfg(enable_shadows=True, enable_ambient_lighting=True, enable_textures=True)
    added = False
    for name, (camera, visualizer_cfg) in tracking_camera_cfgs(sim_cfg).items():
        existing = getattr(scene_cfg, name, None)
        if existing is not None:
            # a launch that already added this camera, for a config reused across launches
            if isinstance(existing, CameraCfg) and existing.prim_path == camera.prim_path:
                continue
            raise ValueError(f"Scene already has an entry named {name!r}; give the tracking camera another prim_path.")
        # the visualizer's background, so the camera's sky matches its viewport
        setattr(
            scene_cfg,
            name,
            make_scene_camera_cfg(camera, camera.renderer_cfg or renderer_cfg, visualizer_cfg.background_color),
        )
        added = True
        num_envs, (width, height) = scene_cfg.num_envs, camera.resolution
        # the count may still be unset, or changed after launch
        gib = num_envs * width * height * 4 / 2**30 if isinstance(num_envs, int) else 0.0
        if gib > 4.0:
            logger.warning(
                "Tracking camera %r allocates about %.0f GiB of image buffers for %d environments, because the scene "
                "renders one image per environment. Use fewer environments or a smaller TrackingCameraCfg.resolution.",
                name,
                gib,
                num_envs,
            )
    return added


class TrackingCameraUpdater:
    """Places a scene camera behind a tracked asset, following its position and, optionally, its yaw."""

    def __init__(self, cfg: TrackingCameraCfg, camera: Camera, scene: InteractiveScene) -> None:
        self.cfg = cfg
        self.camera = camera
        asset_name, _, self._body_name = cfg.track_path.partition("/")
        try:
            self._asset = scene[asset_name]
        except KeyError as exc:
            raise ValueError(f"track_path refers to an unknown scene asset: {asset_name!r}.") from exc
        self._body_index: int | None = None
        self._yaw: torch.Tensor | None = None
        # an eager scene captures cameras before they move, so a move must request fresh pixels
        self._eager = not scene.cfg.lazy_sensor_update

    def update(self, env_ids: list[int], dt: float) -> None:
        """Move the cameras of *env_ids* to the asset's pose after *dt* [s] of simulation, once both are ready."""
        asset, cfg, camera = self._asset, self.cfg, self.camera
        if not (asset.is_initialized and camera.is_initialized) or not env_ids:
            return
        data = asset.data
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=camera.device)
        if self._body_name:
            if self._body_index is None:
                body_ids, _ = asset.find_bodies(self._body_name)
                if len(body_ids) != 1:
                    raise ValueError(f"track_path must match exactly one body: {cfg.track_path!r}.")
                self._body_index = body_ids[0]
            position = data.body_pos_w.torch[ids, self._body_index]
            quat = data.body_quat_w.torch[ids, self._body_index]
        else:
            position, quat = data.root_pos_w.torch[ids], data.root_quat_w.torch[ids]
        eye = torch.tensor(cfg.eye, device=position.device)
        target = torch.tensor(cfg.lookat, device=position.device)
        if cfg.follow_heading:
            yaw = self._filtered_yaw(ids, quat, dt)
            cos, sin = yaw.cos(), yaw.sin()
            eye, target = (
                torch.stack((cos * o[0] - sin * o[1], sin * o[0] + cos * o[1], o[2].expand_as(cos)), dim=-1)
                for o in (eye, target)
            )
        camera.set_world_poses_from_view(position + eye, position + target, env_ids=ids)
        if self._eager:
            camera.update(0.0, force_recompute=True)

    def _filtered_yaw(self, ids: torch.Tensor, quat: torch.Tensor, dt: float) -> torch.Tensor:
        """Return the yaw of *quat* filtered toward along the shortest rotation, the first sample unfiltered."""
        x, y, z, w = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        tau = self.cfg.heading_smoothing_time_constant
        if self._yaw is None:
            self._yaw = torch.full((self._asset.num_instances,), math.nan, device=yaw.device)
        previous = self._yaw[ids]
        if tau > 0.0:
            delta = torch.atan2((yaw - previous).sin(), (yaw - previous).cos())
            yaw = torch.where(previous.isnan(), yaw, previous + -math.expm1(-dt / tau) * delta)
        self._yaw[ids] = yaw
        return yaw
