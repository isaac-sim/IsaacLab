# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg


class RayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `meshes` dictionary.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()
        # Dictionary to hold warp mesh info. Keys are mesh prim patterns.
        self.meshes: dict[str, list[dict]] = {}

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
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
        # resample the drift
        self.drift[env_ids] = self.drift[env_ids].uniform_(*self.cfg.drift_range)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        """
        For each mesh in the configuration, search for matching prims.
        If a prim’s type is not "Mesh" or "Plane", search its children for one that is.
        Then extract the geometry (or create a plane), build a warp mesh,
        and create an XFormPrim for dynamic updates.
        """

        self.meshes = {}
        # read prims to ray-cast
        for pattern in self.cfg.mesh_prim_paths:
            prims = sim_utils.find_matching_prims(pattern)
            if len(prims) == 0:
                omni.log.warn(f"No prims found for pattern: {pattern}")
                continue
            mesh_info_list = []
            for prim in prims:
                prim_type = prim.GetTypeName()
                # if the prim is not directly a Mesh or Plane, search its children for a valid type.
                if prim_type not in ["Mesh", "Plane"]:
                    child = sim_utils.get_first_matching_child_prim(
                        prim.GetPath().pathString, lambda p: p.GetTypeName() in ["Mesh", "Plane"]
                    )
                    if child is not None:
                        prim = child
                        prim_type = prim.GetTypeName()
                    else:
                        omni.log.warn(f"Prim {prim.GetPath()} does not contain a valid Mesh or Plane child.")
                        continue
                # process the prim based on its type.
                if prim_type == "Plane":
                    # create an infinite plane mesh.
                    mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                    base_points = mesh.vertices
                    base_indices = mesh.faces
                    wp_mesh = convert_to_warp_mesh(base_points, base_indices, device=self.device)
                    is_plane = True
                elif prim_type == "Mesh":
                    # read vertices and face indices from the mesh prim.
                    mesh_prim = UsdGeom.Mesh(prim)
                    base_points = np.asarray(mesh_prim.GetPointsAttr().Get())
                    base_indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                    wp_mesh = convert_to_warp_mesh(base_points, base_indices, device=self.device)
                    is_plane = False
                else:
                    omni.log.warn(f"Unsupported prim type '{prim_type}' at {prim.GetPath()}")
                    continue
                # create an XFormPrim to allow dynamic updates.
                xform_view = XFormPrim(prim.GetPath().pathString, reset_xform_properties=False)
                mesh_info = {
                    "prim_path": prim.GetPath().pathString,
                    "xform_view": xform_view,
                    "base_points": base_points,
                    "indices": base_indices,
                    "warp_mesh": wp_mesh,
                    "is_plane": is_plane,
                }
                mesh_info_list.append(mesh_info)
            if len(mesh_info_list) > 0:
                self.meshes[pattern] = mesh_info_list
        if len(self.meshes) == 0:
            raise RuntimeError(f"No valid mesh prims found for ray casting. Patterns: {self.cfg.mesh_prim_paths}")

    def _initialize_rays_impl(self):
        # compute ray starts and directions
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
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        # initialize buffers for storing the closest hit for each ray
        final_hits = torch.full((self._view.count, self.num_rays, 3), float("inf"), device=self.device)
        final_distances = torch.full((self._view.count, self.num_rays), float("inf"), device=self.device)

        # loop over each mesh and its corresponding mesh infos
        for pattern, mesh_info_list in self.meshes.items():
            for mesh_info in mesh_info_list:
                mesh_view = mesh_info["xform_view"]
                indices = torch.arange(mesh_view.count, device=self.device)
                # get the current world transforms for the mesh prim (for dynamic updates)
                mesh_pos, mesh_quat = mesh_view.get_world_poses(indices)
                base_points = torch.tensor(mesh_info["base_points"], device=self.device, dtype=torch.float32)
                # vectorize the transformation by applying the current rotation and translation to all base points using broadcasting
                new_points = quat_apply(
                    mesh_quat, base_points.unsqueeze(0).repeat(mesh_view.count, 1, 1)
                ) + mesh_pos.unsqueeze(1)
                # flatten the transformed points
                new_points_np = new_points.cpu().numpy().reshape(-1, 3)
                # create a new warp mesh with updated vertex positions for dynamic collision detection
                new_wp_mesh = convert_to_warp_mesh(new_points_np, mesh_info["indices"], device=self.device)
                # update the warp mesh in the mesh info dictionary for future use
                mesh_info["warp_mesh"] = new_wp_mesh
                # perform ray casting on the updated warp mesh
                hit, distance, _, _ = raycast_mesh(
                    ray_starts_w,
                    ray_directions_w,
                    max_dist=self.cfg.max_distance,
                    mesh=new_wp_mesh,
                    return_distance=True,
                )
                # use a mask to determine which rays hit this mesh closer than previous hits.
                mask = distance < final_distances
                final_distances[mask] = distance[mask]
                final_hits[mask] = hit[mask]

        # update the ray hit data for the specified environments
        self._data.ray_hits_w[env_ids] = final_hits[env_ids]

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
        # show ray hit positions
        self.ray_visualizer.visualize(self._data.ray_hits_w.view(-1, 3))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._view = None
