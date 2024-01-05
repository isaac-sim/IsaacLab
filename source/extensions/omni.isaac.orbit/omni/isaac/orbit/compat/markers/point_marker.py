# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A class to coordinate groups of visual markers (loaded from USD)."""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from pxr import Gf, UsdGeom


class PointMarker:
    """A class to coordinate groups of visual sphere markers for goal-conditioned tasks.

    This class allows visualization of multiple spheres. These can be used to represent a
    goal-conditioned task. For instance, if a robot is performing a task of reaching a target, the
    class can be used to display a red sphere when the target is far away and a green sphere when
    the target is achieved. Otherwise, the class can be used to display spheres, for example, to
    mark contact points.

    The class uses `UsdGeom.PointInstancer`_ for efficient handling of multiple markers in the stage.
    It creates two spherical markers of different colors. Based on the indices provided the referenced
    marker is activated:

    - :obj:`0` corresponds to unachieved target (red sphere).
    - :obj:`1` corresponds to achieved target (green sphere).

    Usage:
        To create 24 point target markers of radius 0.2 and show them as achieved targets:

        .. code-block:: python

            from omni.isaac.orbit.compat.markers import PointMarker

            # create a point marker
            marker = PointMarker("/World/Visuals/goal", 24, radius=0.2)

            # set position of the marker
            marker_positions = np.random.uniform(-1.0, 1.0, (24, 3))
            marker.set_world_poses(marker_positions)
            # set status of the marker to show achieved targets
            marker.set_status([1] * 24)

    .. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html

    """

    def __init__(self, prim_path: str, count: int, radius: float = 1.0):
        """Initialize the class.

        Args:
            prim_path: The prim path where the PointInstancer will be created.
            count: The number of marker duplicates to create.
            radius: The radius of the sphere. Defaults to 1.0.

        Raises:
            ValueError: When a prim already exists at the :obj:`prim_path` and it is not a :class:`UsdGeom.PointInstancer`.
        """
        # check inputs
        stage = stage_utils.get_current_stage()
        # -- prim path
        if prim_utils.is_prim_path_valid(prim_path):
            prim = prim_utils.get_prim_at_path(prim_path)
            if not prim.IsA(UsdGeom.PointInstancer):
                raise ValueError(f"The prim at path {prim_path} cannot be parsed as a `PointInstancer` object")
            self._instancer_manager = UsdGeom.PointInstancer(prim)
        else:
            self._instancer_manager = UsdGeom.PointInstancer.Define(stage, prim_path)
        # store inputs
        self.prim_path = prim_path
        self.count = count
        self._radius = radius

        # create manager for handling instancing of frame markers
        self._instancer_manager = UsdGeom.PointInstancer.Define(stage, prim_path)
        # create a child prim for the marker
        # -- target not achieved
        prim = prim_utils.create_prim(f"{prim_path}/target_far", "Sphere", attributes={"radius": self._radius})
        geom = UsdGeom.Sphere(prim)
        geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
        # -- target achieved
        prim = prim_utils.create_prim(f"{prim_path}/target_close", "Sphere", attributes={"radius": self._radius})
        geom = UsdGeom.Sphere(prim)
        geom.GetDisplayColorAttr().Set([(0.0, 1.0, 0.0)])
        # -- target invisible
        prim = prim_utils.create_prim(f"{prim_path}/target_invisible", "Sphere", attributes={"radius": self._radius})
        geom = UsdGeom.Sphere(prim)
        geom.GetDisplayColorAttr().Set([(0.0, 0.0, 1.0)])
        prim_utils.set_prim_visibility(prim, visible=False)
        # add child reference to point instancer
        relation_manager = self._instancer_manager.GetPrototypesRel()
        relation_manager.AddTarget(f"{prim_path}/target_far")  # target index: 0
        relation_manager.AddTarget(f"{prim_path}/target_close")  # target index: 1
        relation_manager.AddTarget(f"{prim_path}/target_invisible")  # target index: 2

        # buffers for storing data in pixar Gf format
        # FUTURE: Make them very far away from the scene?
        self._proto_indices = [0] * self.count
        self._gf_positions = [Gf.Vec3f() for _ in range(self.count)]
        self._gf_orientations = [Gf.Quath() for _ in range(self.count)]
        # FUTURE: add option to set scales

        # specify that all initial prims are related to same geometry
        self._instancer_manager.GetProtoIndicesAttr().Set(self._proto_indices)
        # set initial positions of the targets
        self._instancer_manager.GetPositionsAttr().Set(self._gf_positions)
        self._instancer_manager.GetOrientationsAttr().Set(self._gf_orientations)

    def set_visibility(self, visible: bool):
        """Sets the visibility of the markers.

        The method does this through the USD API.

        Args:
            visible: flag to set the visibility.
        """
        imageable = UsdGeom.Imageable(self._instancer_manager)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def set_world_poses(
        self,
        positions: np.ndarray | torch.Tensor | None = None,
        orientations: np.ndarray | torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Update marker poses in the simulation world frame.

        Args:
            positions:
                Positions in the world frame. Shape is (M, 3). Defaults to None, which means left unchanged.
            orientations:
                Quaternion orientations (w, x, y, z) in the world frame of the prims. Shape is (M, 4).
                Defaults to None, which means left unchanged.
            indices: Indices to specify which alter poses.
                Shape is (M,), where M <= total number of markers. Defaults to None (i.e: all markers).
        """
        # resolve inputs
        if positions is not None:
            positions = positions.tolist()
        if orientations is not None:
            orientations = orientations.tolist()
        if indices is None:
            indices = range(self.count)
        # change marker locations
        for i, marker_index in enumerate(indices):
            if positions is not None:
                self._gf_positions[marker_index][:] = positions[i]
            if orientations is not None:
                self._gf_orientations[marker_index].SetReal(orientations[i][0])
                self._gf_orientations[marker_index].SetImaginary(orientations[i][1:])
        # apply to instance manager
        self._instancer_manager.GetPositionsAttr().Set(self._gf_positions)
        self._instancer_manager.GetOrientationsAttr().Set(self._gf_orientations)

    def set_status(self, status: list[int] | np.ndarray | torch.Tensor, indices: Sequence[int] | None = None):
        """Updates the marker activated by the instance manager.

        Args:
            status: Decides which prototype marker to visualize. Shape is (M)
            indices: Indices to specify which alter poses.
                Shape is (M,), where M <= total number of markers. Defaults to None (i.e: all markers).
        """
        # default values
        if indices is None:
            indices = range(self.count)
        # resolve input
        if status is not list:
            proto_indices = status.tolist()
        else:
            proto_indices = status
        # change marker locations
        for i, marker_index in enumerate(indices):
            self._proto_indices[marker_index] = int(proto_indices[i])
        # apply to instance manager
        self._instancer_manager.GetProtoIndicesAttr().Set(self._proto_indices)
