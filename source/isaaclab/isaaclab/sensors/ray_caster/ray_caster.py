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

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.views import XformPrimView
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_cast_utils import obtain_world_pose_from_view
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg

# import logger
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

    # Class variables to share meshes across instances
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
        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
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

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        else:
            num_envs_ids = len(env_ids)
        # resample the drift
        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)
        # resample the height drift
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
        # obtain global simulation view

        self._physics_sim_view = sim_utils.SimulationContext.instance().physics_manager.get_physics_sim_view()
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            available_prims = ",".join([str(p.GetPath()) for p in sim_utils.get_current_stage().Traverse()])
            raise RuntimeError(
                f"Failed to find a prim at path expression: {self.cfg.prim_path}. Available prims: {available_prims}"
            )

        self._view, self._offset = self._obtain_trackable_prim_view(self.cfg.prim_path)

        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        # check number of mesh prims provided
        if len(self.cfg.mesh_prim_paths) != 1:
            raise NotImplementedError(
                f"RayCaster currently only supports one mesh prim. Received: {len(self.cfg.mesh_prim_paths)}"
            )

        # read prims to ray-cast
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # check if mesh already casted into warp mesh
            if mesh_prim_path in RayCaster.meshes:
                continue

            # check if the prim is a plane - handle PhysX plane as a special case
            # if a plane exists then we need to create an infinite mesh that is a plane
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            # if we did not find a plane then we need to read the mesh
            if mesh_prim is None:
                # obtain the mesh prim
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                # check if valid
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
                # cast into UsdGeomMesh
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                # read the vertices and faces
                points = np.asarray(mesh_prim.GetPointsAttr().Get())
                # Get world transform using pure USD (UsdGeom.Xformable)
                xformable = UsdGeom.Xformable(mesh_prim)
                world_transform: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                transform_matrix = np.array(world_transform).T
                points = np.matmul(points, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                # print info
                logger.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                # print info
                logger.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
            # add the warp mesh to the list
            RayCaster.meshes[mesh_prim_path] = wp_mesh

        # throw an error if no meshes are found
        if all([mesh_prim_path not in RayCaster.meshes for mesh_prim_path in self.cfg.mesh_prim_paths]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _initialize_rays_impl(self):
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)
        # apply offset transformation to the rays
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos
        # repeat the rays for each sensor
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self.device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)

    def _update_ray_infos(self, env_ids: Sequence[int]):
        """Updates the ray information buffers."""

        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset[0][env_ids], self._offset[1][env_ids]
        )
        # apply drift to ray starting position in world frame
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # check if user provided attach_yaw_only flag
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' property will be deprecated in a future release."
                " Please use the parameter 'ray_alignment' instead."
            )
            # set ray alignment to yaw
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            # log the warning
            logger.warning(msg)
        # ray cast based on the sensor poses
        if self.cfg.ray_alignment == "world":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            # no rotation is considered and directions are not rotated
            ray_starts_w = self.ray_starts[env_ids]
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_ids)

        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            max_dist=self.cfg.max_distance,
            mesh=RayCaster.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]

        # apply vertical drift to ray starting position in ray caster frame
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self._data.ray_hits_w is None:
            return
        # remove possible inf values
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]

        self.ray_visualizer.visualize(viz_points)

    """
    Internal Helpers.
    """

    def _obtain_trackable_prim_view(
        self, target_prim_path: str
    ) -> tuple[XformPrimView | any, tuple[torch.Tensor, torch.Tensor]]:
        """Obtain a prim view that can be used to track the pose of the parget prim.

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
            # TODO: Need to handle the case where API is present but it is disabled
            if current_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim_view = self._physics_sim_view.create_articulation_view(current_path_expr.replace(".*", "*"))
                logger.info(f"Created articulation view for mesh prim at path: {target_prim_path}")
                break

            # TODO: Need to handle the case where API is present but it is disabled
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

            # switch the current prim to the parent prim
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
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None

    def __del__(self):
        RayCaster._instance_count -= 1
        if RayCaster._instance_count == 0:
            RayCaster.meshes.clear()
