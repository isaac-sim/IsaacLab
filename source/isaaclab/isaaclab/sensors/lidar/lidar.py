# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tiled OVRTX lidar sensor.

The host USD runtime cannot load ovrtx's ``omniSensor`` schema plugin, so the prim is authored as
an ``OmniLidar`` with the WPM attributes and schema names written explicitly. ovrtx reads those
attributes from the exported stage. An unregistered ``OmniLidar`` type is not Xformable, so the
pose written to the renderer is the configured offset composed with the parent prim's world pose.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from isaaclab.sim.utils.prims import clone, get_current_stage
from isaaclab.sim.views import FrameView
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray

from ..sensor_base import SensorBase
from .lidar_data import LidarData

if TYPE_CHECKING:
    from .lidar_cfg import LidarCfg

_PATTERN = "p001"
_API_SCHEMAS = ("OmniSensorGenericLidarWpmAPI", f"OmniSensorGenericLidarWpmFiringPatternAPI:{_PATTERN}")


def _set_attr(prim, name: str, type_name, value) -> None:
    """Create an attribute and set its value. The schema does not need to be registered."""
    prim.CreateAttribute(name, type_name).Set(value)


def _author_lidar(prim, cfg: LidarCfg) -> None:
    """Write the WPM lidar schema and the solid-state firing pattern onto ``prim``."""
    schemas = Sdf.TokenListOp()
    schemas.prependedItems = list(_API_SCHEMAS)
    prim.SetMetadata("apiSchemas", schemas)
    prefix = "omni:sensor:LidarWpm"
    azimuths = [float(value) for value in cfg.azimuth_deg]
    _set_attr(prim, "omni:sensor:modelName", Sdf.ValueTypeNames.String, "LidarWpm")
    _set_attr(prim, "cameraSensorType", Sdf.ValueTypeNames.Token, "lidar")
    _set_attr(prim, "visibility", Sdf.ValueTypeNames.Token, "invisible")
    _set_attr(prim, "omni:sensor:frameRate", Sdf.ValueTypeNames.Double2, Gf.Vec2d(10.0, 1.0))
    _set_attr(prim, f"{prefix}:scanType", Sdf.ValueTypeNames.Token, "solidState")
    _set_attr(prim, f"{prefix}:rayType", Sdf.ValueTypeNames.Token, "idealized")
    _set_attr(prim, f"{prefix}:elementsCoordsType", Sdf.ValueTypeNames.Token, cfg.coords_type)
    _set_attr(prim, f"{prefix}:outputFrameOfReference", Sdf.ValueTypeNames.Token, cfg.frame_of_reference)
    _set_attr(prim, f"{prefix}:instantLidar", Sdf.ValueTypeNames.Bool, cfg.instant_lidar)
    _set_attr(prim, f"{prefix}:includeInvalidPoints", Sdf.ValueTypeNames.Bool, cfg.include_invalid_points)
    _set_attr(prim, f"{prefix}:maxReturns", Sdf.ValueTypeNames.UInt, int(cfg.max_returns))
    if cfg.far_range is not None:
        _set_attr(prim, f"{prefix}:farRangeM", Sdf.ValueTypeNames.Float, float(cfg.far_range))
    _set_attr(prim, f"{prefix}:tiledSubSampling", Sdf.ValueTypeNames.UInt, int(cfg.tiled_subsampling))
    # The solid-state tick is placed at validStartAzimuthDeg, and each pattern azimuth is an offset
    # from that tick. Keep the tick on the boresight so a configured azimuth of 0 looks along the
    # sensor's forward axis.
    _set_attr(prim, f"{prefix}:validStartAzimuthDeg", Sdf.ValueTypeNames.Float, 0.0)
    _set_attr(prim, f"{prefix}:validEndAzimuthDeg", Sdf.ValueTypeNames.Float, 360.0)
    _set_attr(prim, f"{prefix}:rotPatternFiringResDeg", Sdf.ValueTypeNames.Float, 1.0)
    _set_attr(prim, f"{prefix}:minDetectableReflectance", Sdf.ValueTypeNames.Float, 0.0)
    _set_attr(prim, f"{prefix}:detectorNoiseVariance", Sdf.ValueTypeNames.Float, 0.0)
    for name in (
        "jitterAzimuthStdDevDeg",
        "jitterElevationStdDevDeg",
        "offsetAzimuthStdDevDeg",
        "offsetElevationStdDevDeg",
    ):
        _set_attr(prim, f"{prefix}:{name}", Sdf.ValueTypeNames.Float, 0.0)
    pattern = f"{prefix}:firingPattern:{_PATTERN}"
    _set_attr(prim, f"{pattern}:azimuthDeg", Sdf.ValueTypeNames.FloatArray, Vt.FloatArray(azimuths))
    _set_attr(
        prim,
        f"{pattern}:elevationDeg",
        Sdf.ValueTypeNames.FloatArray,
        Vt.FloatArray([float(value) for value in cfg.elevation_deg]),
    )
    _set_attr(
        prim,
        f"{pattern}:fireTimeNs",
        Sdf.ValueTypeNames.UIntArray,
        Vt.UIntArray([int(value) for value in cfg.fire_time_ns]),
    )
    _set_attr(
        prim,
        f"{pattern}:channelId",
        Sdf.ValueTypeNames.UIntArray,
        Vt.UIntArray([int(value) for value in cfg.channel_id]),
    )
    _set_attr(prim, f"{pattern}:rangeId", Sdf.ValueTypeNames.UIntArray, Vt.UIntArray())
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cfg.offset.pos))
    x, y, z, w = cfg.offset.rot
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


@wp.kernel
def _clear_lidar_returns_kernel(
    env_mask: wp.array(dtype=wp.bool),
    points: wp.array2d(dtype=wp.vec3f),
    intensity: wp.array2d(dtype=wp.float32),
    valid: wp.array2d(dtype=wp.bool),
):
    """Clear lidar returns for environments selected by ``env_mask``."""
    env_id, ray_id = wp.tid()
    if not env_mask[env_id]:
        return
    points[env_id, ray_id] = wp.vec3f(wp.inf, wp.inf, wp.inf)
    intensity[env_id, ray_id] = 0.0
    valid[env_id, ray_id] = False


@clone
def spawn_lidar(prim_path: str, cfg: LidarCfg) -> Usd.Prim:
    """Spawn one ``OmniLidar`` prim. The clone decorator replicates it across matching parents.

    The prim is authored with explicit WPM attributes, which ovrtx reads without a registered schema.
    """
    stage = get_current_stage()
    prim = stage.DefinePrim(prim_path, "OmniLidar")
    if not prim.IsValid():
        raise RuntimeError(f"Failed to create an OmniLidar prim at '{prim_path}'.")
    _author_lidar(prim, cfg)
    return prim


class Lidar(SensorBase):
    """Tiled solid-state lidar captured by an OVRTX renderer.

    Each environment contributes one prim. A capture submits those prims as tiles of one lidar
    render product, in environment order, and writes the split point cloud into :attr:`data`.
    When the scene also has cameras on the same renderer, one renderer step produces both.
    """

    cfg: LidarCfg

    def __init__(self, cfg: LidarCfg):
        """Spawn the lidar prims and register the shared renderer.

        Args:
            cfg: Lidar configuration. ``renderer_cfg`` must select the OVRTX renderer.

        Raises:
            ValueError: If ``renderer_cfg`` is not an OVRTX renderer.
        """
        if getattr(cfg.renderer_cfg, "renderer_type", None) != "ovrtx":
            renderer = getattr(cfg.renderer_cfg, "renderer_type", type(cfg.renderer_cfg).__name__)
            raise ValueError(f"Lidar requires an OVRTX renderer, got {renderer!r}.")
        super().__init__(cfg)
        self._data: LidarData | None = None
        self._render_data = None
        self._pose_update_registered = False
        self._local_pos: torch.Tensor | None = None
        self._local_quat: torch.Tensor | None = None
        self._pos_w: ProxyArray | None = None
        self._quat_w: ProxyArray | None = None
        self._parent_view: FrameView | None = None
        self._fallback_origins: torch.Tensor | None = None
        self._fallback_quat: torch.Tensor | None = None
        self._force_capture = False

        from isaaclab import sim as sim_utils

        spawn = self.cfg.spawn
        if spawn is None:
            if not sim_utils.find_matching_prims(self.cfg.prim_path):
                raise RuntimeError(f"Lidar spawn is None, but no lidar prim was found at {self.cfg.prim_path!r}.")
        else:
            spawn_target = spawn.spawn_path or self.cfg.prim_path
            if sim_utils.find_first_matching_prim(spawn_target) is None:
                spawn_lidar(spawn_target, self.cfg)
            if not sim_utils.find_matching_prims(spawn_target):
                raise RuntimeError(f"Could not find a lidar prim at {spawn_target!r}.")
        sim_ctx = sim_utils.SimulationContext.instance()
        sim_ctx.require_visual_shapes()
        self._renderer = sim_ctx.get_or_create_backend(self.cfg.renderer_cfg)

    def __del__(self, _sys=sys):
        """Release the render product. Skipped while the interpreter is shutting down."""
        if _sys.is_finalizing() or _sys.meta_path is None:
            return
        super().__del__()
        parent_view = getattr(self, "_parent_view", None)
        if parent_view is not None:
            parent_view.close()
            self._parent_view = None
        renderer = getattr(self, "_renderer", None)
        if renderer is not None and hasattr(renderer, "cleanup_lidar"):
            renderer.cleanup_lidar(getattr(self, "_render_data", None))

    @property
    def data(self) -> LidarData:
        """Latest lidar returns. Reading this captures the sensor when its buffers are due."""
        self._update_outdated_buffers()
        return self._data

    def update(self, dt: float, force_recompute: bool = False):
        """Advance the sensor clock.

        Args:
            dt: Time elapsed since the previous sensor update, in seconds.
            force_recompute: When True, drop the shared capture cache and render again in this step.
        """
        self._force_capture = force_recompute
        try:
            if force_recompute and self._is_initialized:
                # A repeated read in this physics step must render again. The base update only
                # captures when the buffers are dirty.
                self._data_dirty = True
                invalidate = getattr(self._renderer, "invalidate_sensor_capture", None)
                if invalidate is not None:
                    invalidate()
            super().update(dt, force_recompute=force_recompute)
        finally:
            self._force_capture = False

    def reset(self, env_ids=None, env_mask=None) -> None:
        """Clear returns for the reset environments and drop the shared capture cache."""
        reset_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(env_mask=reset_mask)
        if self._data is None:
            return
        wp.launch(
            _clear_lidar_returns_kernel,
            dim=(self._num_envs, self._data.points.warp.shape[1]),
            inputs=[reset_mask, self._data.points.warp, self._data.intensity.warp, self._data.valid.warp],
            device=self._device,
        )
        invalidate = getattr(self._renderer, "invalidate_sensor_capture", None)
        if invalidate is not None:
            invalidate()

    def _initialize_impl(self):
        """Allocate the return buffers and register the tiled lidar product."""
        super()._initialize_impl()
        from isaaclab import sim as sim_utils

        num_rays = len(self.cfg.azimuth_deg) * self.cfg.max_returns
        self._data = LidarData(self._num_envs, num_rays, self._device)
        self._local_pos = torch.tensor(self.cfg.offset.pos, dtype=torch.float32, device=self._device)
        self._local_quat = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device=self._device)
        self._pos_w = ProxyArray(wp.zeros(self._num_envs, dtype=wp.vec3f, device=self._device))
        self._quat_w = ProxyArray(wp.zeros(self._num_envs, dtype=wp.quatf, device=self._device))
        self._bind_parent_pose()
        self._write_world_poses()

        provider = self._sim.get_scene_data_provider()
        self._sim.render_context.ensure_prepare_stage(self.stage, provider.num_envs)
        lidar_paths = [str(prim.GetPath()) for prim in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage)]
        if not lidar_paths:
            raise RuntimeError(f"Lidar prim '{self.cfg.prim_path}' was not found after the scene was cloned.")
        self._render_data = self._renderer.create_lidar_render_data(
            prim_path=self.cfg.prim_path,
            lidar_paths=lidar_paths,
            num_envs=self._num_envs,
            num_rays=num_rays,
            points=self._data.points,
            intensity=self._data.intensity,
            valid=self._data.valid,
            sample_dt=self._capture_interval(),
        )
        register = getattr(self._renderer, "register_pose_update", None)
        if register is not None and not self._pose_update_registered:
            register(self._push_renderer_pose)
            self._pose_update_registered = True
        self._push_renderer_pose()

    def _update_buffers_impl(self, env_mask: wp.array):
        """Capture every tile when any environment is due. A tiled product cannot render one tile."""
        if self.cfg.update_period > 0.0 and not bool(env_mask.numpy().any()):
            return
        self._renderer.render_lidars(self._capture_interval(), force=self._force_capture)

    def _push_renderer_pose(self) -> None:
        """Write the current world pose of every tile. Safe to call more than once per step."""
        if self._render_data is None or self._pos_w is None:
            return
        self._write_world_poses()
        self._renderer.update_lidar(self._render_data, self._pos_w, self._quat_w)

    def _capture_interval(self) -> float:
        """Return the lidar sampling interval, in seconds.

        A positive ``update_period`` is that interval. Otherwise the interval is the simulation
        step, which is also the interval used when a camera shares this renderer.
        """
        if self.cfg.update_period > 0.0:
            return float(self.cfg.update_period)
        return float(self._sim_physics_dt)

    def _write_world_poses(self) -> None:
        """Compose the configured offset with each parent world pose.

        The parent view follows a translated, rotated, or moving mount. A parent that is not an
        xform uses the clone-plan origins cached at initialization and an identity rotation.
        """
        parent_pos, parent_quat = self._parent_poses()
        pos_w, quat_w = math_utils.combine_frame_transforms(
            parent_pos,
            parent_quat,
            self._local_pos.unsqueeze(0).expand(self._num_envs, -1),
            self._local_quat.unsqueeze(0).expand(self._num_envs, -1),
        )
        self._pos_w.torch.copy_(pos_w)
        self._quat_w.torch.copy_(quat_w)

    def _parent_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one world position and quaternion per environment for the lidar's parent."""
        if self._parent_view is not None:
            position, orientation = self._parent_view.get_world_poses()
            return position.torch, orientation.torch
        return self._fallback_origins, self._fallback_quat

    def _bind_parent_pose(self) -> None:
        """Track the parent prim, or cache fixed origins when that prim is not an xform."""
        from isaaclab import sim as sim_utils

        parent_expr = "/".join(sim_utils.split_path_expr(self.cfg.prim_path)[:-1])
        parents = list(sim_utils.find_matching_prims(parent_expr, self.stage)) if parent_expr else []
        if parents and all(prim.IsA(UsdGeom.Xformable) for prim in parents):
            self._parent_view = FrameView(parent_expr, device=self._device, stage=self.stage)
            if self._parent_view.count != self._num_envs:
                raise RuntimeError(
                    f"Lidar '{self.cfg.prim_path}' has {self._num_envs} instances, but parent"
                    f" '{parent_expr}' resolved {self._parent_view.count} poses."
                )
            return
        self._cache_fallback_origins()

    def _cache_fallback_origins(self) -> None:
        """Store one translation per environment. A global lidar uses the stage origin."""
        if self._is_global:
            origins = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
        else:
            plan = self._sim.get_clone_plan()
            if plan is None or plan.positions is None:
                origins = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
            else:
                origins = torch.as_tensor(plan.positions, dtype=torch.float32, device=self._device)
                if origins.shape != (self._num_envs, 3):
                    raise RuntimeError(
                        f"Lidar '{self.cfg.prim_path}' has {self._num_envs} instances, but the clone plan has"
                        f" origins of shape {tuple(origins.shape)}."
                    )
        self._fallback_origins = origins
        self._fallback_quat = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self._fallback_quat[:, 3] = 1.0
