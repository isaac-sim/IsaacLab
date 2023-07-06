# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from typing import Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from pxr import UsdGeom, UsdPhysics

from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.orbit.utils.math import quat_apply, quat_apply_yaw
from omni.isaac.orbit.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_caster_cfg import RayCasterCfg
from .ray_caster_data import RayCasterData


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

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg (RayCasterCfg): The configuration parameters.
        """
        # store config
        self.cfg = cfg
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()
        # List of meshes to ray-cast
        self.warp_meshes = []

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self._view._regex_prim_paths}': \n"
            f"\tview type : {self._view.__class__}\n"
            f"\tupdate period (s) : {self._update_period}\n"
            f"\tnumber of meshes : {len(self.warp_meshes)}\n"
            f"\tnumber of sensors: {self.view.count}\n"
            f"\tnumber of rays/sensor : {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self.view.count}"
        )

    """
    Properties
    """

    @property
    def data(self) -> RayCasterData:
        """Data related to ray-caster."""
        return self._data

    """
    Operations
    """

    def spawn(self, prim_path: str, *args, **kwargs):
        """Spawns the sensor in the scene.

        Note:
            Ray-caster is a virtual sensor and does not need to be spawned. However,
            this function is required by the base class.
        """
        pass

    def initialize(self, prim_paths_expr: str):
        # check if the prim at path is a articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        prim_view_class = None
        for prim_path in prim_utils.find_matching_prim_paths(prim_paths_expr):
            # get prim at path
            prim = prim_utils.get_prim_at_path(prim_path)
            # check if it is a rigid prim
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim_view_class = ArticulationView
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim_view_class = RigidPrimView
            else:
                prim_view_class = XFormPrimView
                carb.log_warn(f"The prim at path {prim_path} is not a rigid prim! Using XFormPrimView.")
            # break the loop
            break
        # check if prim view class is found
        if prim_view_class is None:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {prim_paths_expr}")
        # create a rigid prim view for the sensor
        self._view = prim_view_class(prim_paths_expr, reset_xform_properties=False)
        self._view.initialize()
        # initialize the base class
        super().initialize(prim_paths_expr)
        # Check that backend is compatible
        if self._backend != "torch":
            raise RuntimeError(f"RayCaster only supports PyTorch backend. Received: {self._backend}.")

        # check number of mesh prims provided
        if len(self.cfg.mesh_prim_paths) != 1:
            raise NotImplementedError(
                f"RayCaster currently only supports one mesh prim. Received: {len(self.cfg.mesh_prim_paths)}"
            )
        # read prims to ray-cast
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # obtain the mesh prim
            mesh_prim = prim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda p: prim_utils.get_prim_type_name(p) == "Mesh"
            )
            # check if valid
            if not prim_utils.is_prim_path_valid(mesh_prim_path):
                raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
            # cast into UsdGeomMesh
            mesh_prim = UsdGeom.Mesh(mesh_prim)
            # read the vertices and faces
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
            wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
            # add the warp mesh to the list
            self.warp_meshes.append(wp_mesh)
        # throw an error if no meshes are found
        if len(self.warp_meshes) == 0:
            raise RuntimeError("No meshes found for ray-casting! Please check the mesh prim paths.")

        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)
        # apply offset transformation to the rays
        offset_pos = torch.tensor(list(self.cfg.pos_offset), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.quat_offset), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos
        # repeat the rays for each sensor
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)

        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

        # visualization of the ray-caster
        prim_path = stage_utils.get_next_free_path("/Visuals/RayCaster")
        self.ray_visualizer = VisualizationMarkers(prim_path, cfg=RAY_CASTER_MARKER_CFG)

    def reset_buffers(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals.

        Args:
            env_ids (Sequence[int], optional): The sensor ids to reset. Defaults to None.
        """
        # reset the timers and counters
        super().reset_buffers(env_ids)
        # force buffer the data -> needed for reset observations
        self._buffer(env_ids)

    def debug_vis(self):
        # visualize the point hits
        if self.cfg.debug_vis:
            # check if prim is visualized
            self.ray_visualizer.visualize(self._data.ray_hits_w.view(-1, 3))

    """
    Implementation.
    """

    def _buffer(self, env_ids: Sequence[int] | None = None):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # obtain the poses of the sensors
        pos_w, quat_w = self._view.get_world_poses(env_ids, clone=False)
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
        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._data.ray_hits_w[env_ids] = raycast_mesh(ray_starts_w, ray_directions_w, self.warp_meshes[0])
