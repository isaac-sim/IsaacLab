# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene cameras that follow an asset, shown in the visualizer streaming view."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .visualizer_cfg import USD_DEFAULT_VERTICAL_APERTURE_MM, TrackedCameraCfg, VisualizerCfg

if TYPE_CHECKING:
    from ..scene import InteractiveScene
    from ..sensors import Camera, CameraCfg


def tracked_camera_cfgs(*visualizer_cfgs) -> dict[str, tuple[TrackedCameraCfg, VisualizerCfg]]:
    """Return the tracked cameras the visualizer configs declare with their visualizer, keyed by sensor name."""
    cameras = {}
    for cfg in visualizer_cfgs:
        for camera in (cfg.cameras or []) if cfg is not None else []:
            if isinstance(camera, TrackedCameraCfg):
                cameras.setdefault(camera.prim_path.rsplit("/", 1)[-1], (camera, cfg))
    return cameras


def make_scene_camera_cfg(cfg: TrackedCameraCfg, renderer_cfg, background_color=None) -> CameraCfg:
    """Return the scene :class:`~isaaclab.sensors.CameraCfg` that realizes *cfg* in every environment.

    The camera starts at the configured offsets, so a camera that tracks nothing stays fixed relative to its
    environment origin. Tracked cameras get their pose from :class:`TrackedCameraUpdater` instead.
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


def add_tracked_cameras(cfg, args: dict) -> None:
    """Add the scene cameras that tracked cameras declare to *cfg*'s scene, if a visualizer will use them.

    A visualizer uses them when ``--visualizer`` selects one or a video recorder records a ``streaming_view``
    source, so runs that display nothing pay nothing. Without ``renderer_cfg``, Newton physics renders them with
    the Newton Warp renderer and other physics with the camera default.

    Args:
        cfg: The launched config; only an environment config with ``sim`` and ``scene`` is changed.
        args: Launcher arguments whose ``visualizer`` selection is already resolved.
    """
    from ..envs.utils.video_recorder_cfg import parse_video_source
    from ..sim import SimulationCfg

    sim_cfg, scene_cfg = getattr(cfg, "sim", None), getattr(cfg, "scene", None)
    if not isinstance(sim_cfg, SimulationCfg) or scene_cfg is None:
        return
    recorded = any(
        parse_video_source(recorder.source)[2] == "streaming_view" for recorder in getattr(cfg, "video_recorders", ())
    )
    if not (args["visualizer"] or recorded):
        return
    renderer_cfg = None
    if "newton" in str(args.get("physics") or type(sim_cfg.physics).__name__).lower():
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        renderer_cfg = NewtonWarpRendererCfg(enable_shadows=True, enable_ambient_lighting=True, enable_textures=True)
    for name, (camera, visualizer_cfg) in tracked_camera_cfgs(
        sim_cfg.default_visualizer_cfg, *sim_cfg.visualizer_cfgs
    ).items():
        if not hasattr(scene_cfg, name):
            # the visualizer's background, so the camera's sky matches its viewport
            scene_camera_cfg = make_scene_camera_cfg(
                camera, camera.renderer_cfg or renderer_cfg, visualizer_cfg.background_color
            )
            setattr(scene_cfg, name, scene_camera_cfg)


class TrackedCameraUpdater:
    """Places a scene camera behind a tracked asset, following its position and, optionally, its yaw."""

    def __init__(self, cfg: TrackedCameraCfg, camera: Camera, scene: InteractiveScene) -> None:
        self.cfg = cfg
        self.camera = camera
        asset_name, _, self._body_name = cfg.track_path.partition("/")
        try:
            self._asset = scene[asset_name]
        except KeyError as exc:
            raise ValueError(f"track_path refers to an unknown scene asset: {asset_name!r}.") from exc
        self._body_index: int | None = None
        self._yaw: torch.Tensor | None = None

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
