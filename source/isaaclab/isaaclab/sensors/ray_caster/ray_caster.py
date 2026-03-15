# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.views import XformPrimView
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh

from ..sensor_base import SensorBase
from .kernels import (
    apply_z_drift_kernel,
    fill_vec3_inf_kernel,
    raycast_mesh_masked_kernel,
    update_ray_caster_kernel,
)
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg

logger = logging.getLogger(__name__)


class RayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    meshes: ClassVar[dict[str, wp.Mesh]] = {}
    """A dictionary to store warp meshes for raycasting, shared across all instances.

    The keys correspond to the prim path for the meshes, and values are the corresponding warp Mesh objects."""
    _instance_count: ClassVar[int] = 0
    """A counter to track the number of RayCaster instances, used to manage class variable lifecycle."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        RayCaster._instance_count += 1
        super().__init__(cfg)
        self._data = RayCasterData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(RayCaster.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # reset the timers and counters
        super().reset(env_ids, env_mask)
        # resolve to indices for torch indexing
        if env_ids is not None:
            num_envs_ids = len(env_ids)
        elif env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
            num_envs_ids = len(env_ids)
        else:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        # resample drift (uses torch views for indexing)
        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)
        # resample the ray cast drift
        range_list = [self.cfg.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        self._physics_sim_view = sim_utils.SimulationContext.instance().physics_manager.get_physics_sim_view()
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            available_prims = ",".join([str(p.GetPath()) for p in sim_utils.get_current_stage().Traverse()])
            raise RuntimeError(
                f"Failed to find a prim at path expression: {self.cfg.prim_path}. Available prims: {available_prims}"
            )

        self._view, self._offset = self._obtain_trackable_prim_view(self.cfg.prim_path)

        # Convert offsets to warp (zero-copy from existing torch tensors)
        self._offset_pos_wp = wp.from_torch(self._offset[0].contiguous(), dtype=wp.vec3f)
        self._offset_quat_wp = wp.from_torch(self._offset[1].contiguous(), dtype=wp.quatf)

        # Handle deprecated attach_yaw_only at init time
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' property will be deprecated in a future release."
                " Please use the parameter 'ray_alignment' instead."
            )
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            logger.warning(msg)
            self.cfg.attach_yaw_only = None

        # Resolve alignment mode to integer constant for kernel dispatch
        alignment_map = {"world": 0, "yaw": 1, "base": 2}
        if self.cfg.ray_alignment not in alignment_map:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")
        self._alignment_mode = alignment_map[self.cfg.ray_alignment]

        self._initialize_warp_meshes()
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        # check number of mesh prims provided
        if len(self.cfg.mesh_prim_paths) != 1:
            raise NotImplementedError(
                f"RayCaster currently only supports one mesh prim. Received: {len(self.cfg.mesh_prim_paths)}"
            )

        # read prims to ray-cast
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            if mesh_prim_path in RayCaster.meshes:
                continue

            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            if mesh_prim is None:
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                points = np.asarray(mesh_prim.GetPointsAttr().Get())
                xformable = UsdGeom.Xformable(mesh_prim)
                world_transform: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                transform_matrix = np.array(world_transform).T
                points = np.matmul(points, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                logger.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                logger.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
            RayCaster.meshes[mesh_prim_path] = wp_mesh

        if all([mesh_prim_path not in RayCaster.meshes for mesh_prim_path in self.cfg.mesh_prim_paths]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _initialize_rays_impl(self):
        # Compute ray starts and directions from pattern (torch, init-time only)
        ray_starts_torch, ray_directions_torch = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(ray_directions_torch)

        # Apply sensor offset rotation/position to local ray pattern
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        ray_directions_torch = math_utils.quat_apply(
            offset_quat.repeat(len(ray_directions_torch), 1), ray_directions_torch
        )
        ray_starts_torch += offset_pos

        # Repeat for each environment
        ray_starts_torch = ray_starts_torch.repeat(self._view.count, 1, 1)
        ray_directions_torch = ray_directions_torch.repeat(self._view.count, 1, 1)

        # Create warp arrays from the init-time torch data
        # The warp arrays own the memory; torch views provide backward-compat indexing
        self._ray_starts_local = wp.from_torch(ray_starts_torch.contiguous(), dtype=wp.vec3f)
        self._ray_directions_local = wp.from_torch(ray_directions_torch.contiguous(), dtype=wp.vec3f)

        # Torch views (same attribute names as before for subclass compatibility)
        self.ray_starts = wp.to_torch(self._ray_starts_local)
        self.ray_directions = wp.to_torch(self._ray_directions_local)

        # Drift buffers (warp-owned, torch views for reset indexing)
        self._drift = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)
        self._ray_cast_drift = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)
        self.drift = wp.to_torch(self._drift)
        self.ray_cast_drift = wp.to_torch(self._ray_cast_drift)

        # World-frame ray buffers
        self._ray_starts_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Torch views for subclass compatibility
        self._ray_starts_w_torch = wp.to_torch(self._ray_starts_w)
        self._ray_directions_w_torch = wp.to_torch(self._ray_directions_w)

        # Data buffers
        self._data.create_buffers(self._view.count, self.num_rays, self._device)

    def _get_view_transforms_wp(self) -> wp.array:
        """Get world transforms from the physics view as a warp array.

        Returns:
            Warp array of ``wp.transformf`` with shape (num_envs,).
        """
        if isinstance(self._view, XformPrimView):
            pos_w, quat_w = self._view.get_world_poses()
            poses = torch.cat([pos_w, quat_w], dim=-1).contiguous()
            return wp.from_torch(poses).view(wp.transformf)
        elif isinstance(self._view, physx.ArticulationView):
            return self._view.get_root_transforms().view(wp.transformf)
        elif isinstance(self._view, physx.RigidBodyView):
            return self._view.get_transforms().view(wp.transformf)
        else:
            raise NotImplementedError(f"Cannot get transforms for view type '{type(self._view)}'.")

    def _update_ray_infos(self, env_mask: wp.array):
        """Updates sensor poses and ray world-frame buffers via a single warp kernel."""
        transforms = self._get_view_transforms_wp()

        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                transforms,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self._drift,
                self._ray_cast_drift,
                self._ray_starts_local,
                self._ray_directions_local,
                self._alignment_mode,
            ],
            outputs=[
                self._data._pos_w,
                self._data._quat_w,
                self._ray_starts_w,
                self._ray_directions_w,
            ],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)

        # Fill ray hits with inf before raycasting
        wp.launch(
            fill_vec3_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, self._data._ray_hits_w, float("inf")],
            device=self._device,
        )

        # Ray-cast against the mesh
        wp.launch(
            raycast_mesh_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                RayCaster.meshes[self.cfg.mesh_prim_paths[0]].id,
                env_mask,
                self._ray_starts_w,
                self._ray_directions_w,
                self._data._ray_hits_w,
                float(self.cfg.max_distance),
            ],
            device=self._device,
        )

        # Apply vertical drift to ray hits
        wp.launch(
            apply_z_drift_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, self._ray_cast_drift, self._data._ray_hits_w],
            device=self._device,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self._data._ray_hits_w is None:
            return
        ray_hits_torch = wp.to_torch(self._data._ray_hits_w)
        viz_points = ray_hits_torch.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        self.ray_visualizer.visualize(viz_points)

    """
    Internal Helpers.
    """

    def _obtain_trackable_prim_view(
        self, target_prim_path: str
    ) -> tuple[XformPrimView | any, tuple[torch.Tensor, torch.Tensor]]:
        """Obtain a prim view that can be used to track the pose of the target prim.

        The target prim path is a regex expression that matches one or more mesh prims. While we can track its
        pose directly using XFormPrim, this is not efficient and can be slow. Instead, we create a prim view
        using the physics simulation view, which provides a more efficient way to track the pose of the mesh prims.

        The function additionally resolves the relative pose between the mesh and its corresponding physics prim.
        This is especially useful if the mesh is not directly parented to the physics prim.

        Args:
            target_prim_path: The target prim path to obtain the prim view for.

        Returns:
            A tuple containing:

            - An XFormPrim or a physics prim view (ArticulationView or RigidBodyView).
            - A tuple containing the positions and orientations of the mesh prims in the physics prim frame.

        """

        mesh_prim = sim_utils.find_first_matching_prim(target_prim_path)
        current_prim = mesh_prim
        current_path_expr = target_prim_path

        prim_view = None

        while prim_view is None:
            if current_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim_view = self._physics_sim_view.create_articulation_view(current_path_expr.replace(".*", "*"))
                logger.info(f"Created articulation view for mesh prim at path: {target_prim_path}")
                break

            if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim_view = self._physics_sim_view.create_rigid_body_view(current_path_expr.replace(".*", "*"))
                logger.info(f"Created rigid body view for mesh prim at path: {target_prim_path}")
                break

            new_root_prim = current_prim.GetParent()
            current_path_expr = current_path_expr.rsplit("/", 1)[0]
            if not new_root_prim.IsValid():
                prim_view = XformPrimView(target_prim_path, device=self._device, stage=self.stage)
                current_path_expr = target_prim_path
                logger.warning(
                    f"The prim at path {target_prim_path} which is used for raycasting is not a physics prim."
                    " Defaulting to XFormPrim. \n The pose of the mesh will most likely not"
                    " be updated correctly when running in headless mode and position lookups will be much slower. \n"
                    " If possible, ensure that the mesh or its parent is a physics prim (rigid body or articulation)."
                )
                break

            current_prim = new_root_prim

        # obtain the relative transforms between target prim and the view prims
        mesh_prims = sim_utils.find_matching_prims(target_prim_path)
        view_prims = sim_utils.find_matching_prims(current_path_expr)
        if len(mesh_prims) != len(view_prims):
            raise RuntimeError(
                f"The number of mesh prims ({len(mesh_prims)}) does not match the number of physics prims"
                f" ({len(view_prims)})Please specify the correct mesh and physics prim paths more"
                " specifically in your target expressions."
            )
        positions = []
        quaternions = []
        for mesh_prim, view_prim in zip(mesh_prims, view_prims):
            pos, orientation = sim_utils.resolve_prim_pose(mesh_prim, view_prim)
            positions.append(torch.tensor(pos, dtype=torch.float32, device=self.device))
            quaternions.append(torch.tensor(orientation, dtype=torch.float32, device=self.device))

        positions = torch.stack(positions).to(device=self.device, dtype=torch.float32)
        quaternions = torch.stack(quaternions).to(device=self.device, dtype=torch.float32)

        return prim_view, (positions, quaternions)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        self._view = None

    def __del__(self):
        RayCaster._instance_count -= 1
        if RayCaster._instance_count == 0:
            RayCaster.meshes.clear()
