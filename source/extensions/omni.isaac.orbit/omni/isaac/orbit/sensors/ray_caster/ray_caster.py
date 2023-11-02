# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, ClassVar, Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils
import warp as wp
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from pxr import UsdGeom, UsdPhysics

from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.orbit.terrains.trimesh.utils import make_plane
from omni.isaac.orbit.utils.math import quat_apply, quat_apply_yaw
from omni.isaac.orbit.utils.warp import convert_to_warp_mesh, raycast_mesh

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
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""
    meshes: ClassVar[dict[str, wp.Mesh]] = {}
    """The warp meshes available for raycasting.

    The keys correspond to the prim path for the meshes, and values are the corresponding warp Mesh objects.

    Note:
           We store a global dictionary of all warp meshes to prevent re-loading the mesh for different ray-cast sensor instances.
    """

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
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
        self.drift[env_ids].uniform_(*self.cfg.drift_range)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        prim_view_class = None
        for prim_path in prim_utils.find_matching_prim_paths(self.cfg.prim_path):
            # get prim at path
            prim = prim_utils.get_prim_at_path(prim_path)
            # check if it is a rigid prim
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim_view_class = ArticulationView
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim_view_class = RigidPrimView
            else:
                prim_view_class = XFormPrimView
                carb.log_warn(f"The prim at path {prim_path} is not a physics prim! Using XFormPrimView.")
            # break the loop
            break
        # check if prim view class is found
        if prim_view_class is None:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")
        # create a rigid prim view for the sensor
        self._view = prim_view_class(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()

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
            mesh_prim = prim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda p: prim_utils.get_prim_type_name(p) == "Plane"
            )
            # if we did not find a plane then we need to read the mesh
            if mesh_prim is None:
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
                # print info
                carb.log_info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
            else:
                mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                # print info
                carb.log_info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
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
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        pos_w, quat_w = self._view.get_world_poses(env_ids, clone=False)
        pos_w += self.drift[env_ids]
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
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            mesh=RayCaster.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                visualizer_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster")
                self.ray_visualizer = VisualizationMarkers(visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # show ray hit positions
        self.ray_visualizer.visualize(self._data.ray_hits_w.view(-1, 3))
