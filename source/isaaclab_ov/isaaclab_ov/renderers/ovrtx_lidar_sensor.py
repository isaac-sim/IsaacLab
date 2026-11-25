# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lidar sensor.

An ``OmniLidar`` prim carries a LidarWpm scan. Those
model attributes are written explicitly because the host USD runtime does not have the
``omniSensor`` schema registered. An unregistered ``OmniLidar`` type is not Xformable, so the pose
written to the renderer composes the local mount pose with the parent prim's world pose.
A renderer that implements lidar capture reads those attributes.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from isaaclab.sensors import LidarData, SensorBase
from isaaclab.sim.utils.prims import clone, get_current_stage
from isaaclab.sim.views import FrameView
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from .ovrtx_lidar_cfg import OVRTXLidarCfg


def _apply_api_schemas(prim, names: tuple[str, ...]) -> None:
    """Record the lidar API schemas. The host runtime does not have them registered."""
    schemas = Sdf.TokenListOp()
    schemas.prependedItems = list(names)
    prim.SetMetadata("apiSchemas", schemas)


def _author_output(prim, prefix: str, cfg: OVRTXLidarCfg, ray_type: str) -> None:
    """Write the LidarWpm output contract."""
    prim.CreateAttribute(f"{prefix}:scanType", Sdf.ValueTypeNames.Token).Set(cfg.scan_type)
    prim.CreateAttribute(f"{prefix}:rayType", Sdf.ValueTypeNames.Token).Set(ray_type)
    prim.CreateAttribute(f"{prefix}:elementsCoordsType", Sdf.ValueTypeNames.Token).Set(cfg.elements_coords_type)
    prim.CreateAttribute(f"{prefix}:outputFrameOfReference", Sdf.ValueTypeNames.Token).Set(
        cfg.output_frame_of_reference
    )
    prim.CreateAttribute(f"{prefix}:instantLidar", Sdf.ValueTypeNames.Bool).Set(cfg.instant_lidar)
    prim.CreateAttribute(f"{prefix}:includeInvalidPoints", Sdf.ValueTypeNames.Bool).Set(cfg.include_invalid_points)
    prim.CreateAttribute(f"{prefix}:maxReturns", Sdf.ValueTypeNames.UInt).Set(cfg.max_returns)
    prim.CreateAttribute(f"{prefix}:farRangeM", Sdf.ValueTypeNames.Float).Set(cfg.far_range_m)
    prim.CreateAttribute(f"{prefix}:validStartAzimuthDeg", Sdf.ValueTypeNames.Float).Set(cfg.valid_start_azimuth_deg)
    prim.CreateAttribute(f"{prefix}:validEndAzimuthDeg", Sdf.ValueTypeNames.Float).Set(cfg.valid_end_azimuth_deg)
    prim.CreateAttribute(f"{prefix}:rotPatternFiringResDeg", Sdf.ValueTypeNames.Float).Set(
        cfg.rot_pattern_firing_res_deg
    )


def _author_pattern(prim, prefix: str, cfg: OVRTXLidarCfg) -> None:
    """Write one firing pattern or emitter state from the configured ray arrays."""
    prim.CreateAttribute(f"{prefix}:azimuthDeg", Sdf.ValueTypeNames.FloatArray).Set(Vt.FloatArray(cfg.azimuth_deg))
    prim.CreateAttribute(f"{prefix}:elevationDeg", Sdf.ValueTypeNames.FloatArray).Set(Vt.FloatArray(cfg.elevation_deg))
    prim.CreateAttribute(f"{prefix}:fireTimeNs", Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray(cfg.fire_time_ns))
    prim.CreateAttribute(f"{prefix}:channelId", Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray(cfg.channel_id))


def _author_wpm(prim, cfg: OVRTXLidarCfg) -> None:
    """Write ``LidarWpm`` and its ``p001`` firing pattern."""
    firing_pattern = "p001"
    _apply_api_schemas(
        prim, ("OmniSensorGenericLidarWpmAPI", f"OmniSensorGenericLidarWpmFiringPatternAPI:{firing_pattern}")
    )
    prefix = "omni:sensor:LidarWpm"
    _author_output(prim, prefix, cfg, "idealized")
    prim.CreateAttribute(f"{prefix}:tiledSubSampling", Sdf.ValueTypeNames.UInt).Set(int(cfg.tiled_subsampling))
    prim.CreateAttribute(f"{prefix}:minDetectableReflectance", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute(f"{prefix}:detectorNoiseVariance", Sdf.ValueTypeNames.Float).Set(0.0)
    for name in (
        "jitterAzimuthStdDevDeg",
        "jitterElevationStdDevDeg",
        "offsetAzimuthStdDevDeg",
        "offsetElevationStdDevDeg",
    ):
        prim.CreateAttribute(f"{prefix}:{name}", Sdf.ValueTypeNames.Float).Set(0.0)
    pattern = f"{prefix}:firingPattern:{firing_pattern}"
    _author_pattern(prim, pattern, cfg)
    prim.CreateAttribute(f"{pattern}:rangeId", Sdf.ValueTypeNames.UIntArray).Set(Vt.UIntArray())


@clone
def spawn_lidar(prim_path: str, cfg: OVRTXLidarCfg) -> Usd.Prim:
    """Spawn one ``OmniLidar`` prim. The clone decorator replicates it across matching parents.

    The prim is authored with explicit model attributes because the host runtime does not have the
    ``omniSensor`` schema registered. A renderer that supports lidar reads those attributes.
    """
    stage = get_current_stage()
    prim = stage.DefinePrim(prim_path, "OmniLidar")
    if not prim.IsValid():
        raise RuntimeError(f"Failed to create an OmniLidar prim at '{prim_path}'.")

    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cfg.offset.pos))
    x, y, z, w = cfg.offset.rot
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))

    prim.CreateAttribute("omni:sensor:modelName", Sdf.ValueTypeNames.String).Set("LidarWpm")
    prim.CreateAttribute("cameraSensorType", Sdf.ValueTypeNames.Token).Set("lidar")
    prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
    prim.CreateAttribute("omni:sensor:frameRate", Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(10.0, 1.0))

    _author_wpm(prim, cfg)

    return prim


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


class OVRTXLidar(SensorBase):
    """Lidar sensor captured by one renderer.

    Each environment contributes one LidarWpm prim.
    ``scan_type`` selects a solid-state or rotary scan. A capture submits those prims as
    tiles of one lidar render product, in environment order, and writes the split point cloud into
    :attr:`data`. The buffer length is the full scan: one firing of the pattern for a solid-state
    scan, or one firing at every azimuth step for a rotary scan.
    """

    cfg: OVRTXLidarCfg

    def __init__(self, cfg: OVRTXLidarCfg):
        """Spawn the lidar prims and register the shared renderer.

        Args:
            cfg: Lidar configuration. ``renderer_cfg`` selects the renderer that captures the scan.
        """
        super().__init__(cfg)

        self._data: LidarData | None = None
        self._render_data = None
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
                spawn.func(spawn_target, self.cfg)

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
        if renderer is not None:
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
            force_recompute: When True, capture again even if the sampling period has not expired.
        """
        self._force_capture = force_recompute
        try:
            if force_recompute and self._is_initialized:
                # A repeated read in this physics step must render again. The base update only
                # captures when the buffers are dirty.
                self._data_dirty = True
            super().update(dt, force_recompute=force_recompute)
        finally:
            self._force_capture = False

    def reset(self, env_ids=None, env_mask=None) -> None:
        """Clear returns for the reset environments.

        A global lidar has one buffer row. Scene environment IDs and masks therefore
        clear the whole scan instead of indexing that row as if it were a replicated tile.
        """
        # World lidars have no per-environment rows; any scene reset refreshes the whole lidar.
        if self._is_global:
            env_ids = env_mask = None

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

    def _initialize_impl(self):
        """Allocate the return buffers and register the tiled lidar product."""
        super()._initialize_impl()

        from isaaclab import sim as sim_utils

        num_rays = self.cfg.num_rays()
        self._data = LidarData(self._num_envs, num_rays, self._device)
        self._pos_w = ProxyArray(wp.zeros(self._num_envs, dtype=wp.vec3f, device=self._device))
        self._quat_w = ProxyArray(wp.zeros(self._num_envs, dtype=wp.quatf, device=self._device))
        self._bind_parent_pose()

        provider = self._sim.get_scene_data_provider()
        self._sim.render_context.ensure_prepare_stage(self.stage, provider.num_envs)
        lidar_paths = [str(prim.GetPath()) for prim in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage)]
        if not lidar_paths:
            raise RuntimeError(f"Lidar prim '{self.cfg.prim_path}' was not found after the scene was cloned.")

        self._render_data = self._renderer.create_lidar_render_data(self.cfg, lidar_paths, self._num_envs)
        if self.cfg.spawn is None:
            positions, rotations = self._read_authored_mounts(lidar_paths)
        else:
            positions, rotations = self.cfg.offset.pos, self.cfg.offset.rot
        self._local_pos = torch.tensor(positions, dtype=torch.float32, device=self._device)
        self._local_quat = torch.tensor(rotations, dtype=torch.float32, device=self._device)
        self._push_renderer_pose()

    def _invalidate_initialize_callback(self, event):
        """Release the lidar product so the next initialization does not leave the previous one.

        A hard reset runs initialization again on this sensor. ``__del__`` is too late: replacing
        ``_render_data`` would leave the previous product and its output buffers registered.
        """
        if self._render_data is not None:
            self._renderer.cleanup_lidar(self._render_data)
        self._render_data = None
        super()._invalidate_initialize_callback(event)
        if self._parent_view is not None:
            self._parent_view.close()
            self._parent_view = None

    def _update_buffers_impl(self, env_mask: wp.array):
        """Capture every tile when any environment is due. A tiled product cannot render one tile.

        A forced capture still renders when the sampling period has not expired. ``update`` marks
        the buffers dirty without filling ``_is_outdated``, so the empty mask is not a reason to
        keep the previous scan.
        """
        if self.cfg.update_period > 0.0 and not self._force_capture and not bool(env_mask.numpy().any()):
            return

        self._push_renderer_pose()
        self._sim.render_context.update_scene_state(self._sim.get_physics_step_count())
        self._renderer.render_lidar(self._render_data, self._data)

    def _push_renderer_pose(self) -> None:
        """Write the current world pose of every tile. Safe to call more than once per step."""
        if self._render_data is None or self._pos_w is None:
            return

        self._write_world_poses()
        self._renderer.update_lidar(self._render_data, self._pos_w, self._quat_w)

    def _read_authored_mounts(self, authored_paths: list[str]) -> tuple[list, list]:
        """Read local USD transforms once and order them like the tiled render product."""
        mounts = {}
        for path in authored_paths:
            # OmniLidar may be unregistered in host USD. XformCache then skips its authored xform ops.
            transform = Gf.Transform(UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetLocalTransformation())
            rotation = transform.GetRotation().GetQuat()
            mounts[Sdf.Path(path)] = (tuple(transform.GetTranslation()), (*rotation.imaginary, rotation.real))

        if len(mounts) == 1:
            pos, rot = next(iter(mounts.values()))
            return [pos], [rot]

        # Native copies can exist only in OvRTX. Resolve each missing tile against its source subtree.
        sources = {
            Sdf.Path(target): Sdf.Path(root)
            for root, targets in self._renderer.scene.clone_copies
            for target in targets
        }
        positions, rotations = [], []
        for tile_path in self._render_data.lidar_paths:
            path = Sdf.Path(tile_path)
            if path not in mounts:
                for prefix in reversed(path.GetPrefixes()):
                    if prefix in sources:
                        path = path.ReplacePrefix(prefix, sources[prefix])
                        break
            if path not in mounts:
                raise RuntimeError(f"Cannot resolve the authored lidar mount for {tile_path!r}.")
            pos, rot = mounts[path]
            positions.append(pos)
            rotations.append(rot)
        return positions, rotations

    def _write_world_poses(self) -> None:
        """Compose the local mount pose with each parent world pose.

        The parent view follows a translated, rotated, or moving mount. A parent that is not an
        xform uses the clone-plan origins cached at initialization and an identity rotation.
        """
        parent_pos, parent_quat = self._parent_poses()
        pos_w, quat_w = math_utils.combine_frame_transforms(
            parent_pos,
            parent_quat,
            self._local_pos.expand(self._num_envs, -1),
            self._local_quat.expand(self._num_envs, -1),
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
