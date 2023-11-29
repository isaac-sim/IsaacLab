# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper class to handle visual sphere markers to show ray-casting of height scanner."""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from pxr import Gf, UsdGeom


class HeightScannerMarker:
    """Helper class to handle visual sphere markers to show ray-casting of height scanner.

    The class uses :class:`UsdGeom.PointInstancer` for efficient handling of multiple markers in the stage.
    It creates two spherical markers of different colors. Based on the indices provided the referenced
    marker is activated.

    The status marker (proto-indices) of the point instancer is used to store the following information:

    - :obj:`0` -> ray miss (blue sphere).
    - :obj:`1` -> successful ray hit (red sphere).
    - :obj:`2` -> invisible ray (disabled visualization)

    """

    def __init__(self, prim_path: str, count: int, radius: float = 1.0) -> None:
        """Initialize the class.

        Args:
            prim_path: The prim path of the point instancer.
            count: The number of markers to create.
            radius: The radius of the spherical markers. Defaults to 1.0.

        Raises:
            ValueError: When a prim at the given path exists but is not a valid point instancer.
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
        # TODO: Make this generic marker for all and put inside the `omni.isaac.orbit.marker` directory.
        # create a child prim for the marker
        # -- target missed
        prim = prim_utils.create_prim(f"{prim_path}/point_miss", "Sphere", attributes={"radius": self._radius})
        geom = UsdGeom.Sphere(prim)
        geom.GetDisplayColorAttr().Set([(0.0, 0.0, 1.0)])
        # -- target achieved
        prim = prim_utils.create_prim(f"{prim_path}/point_hit", "Sphere", attributes={"radius": self._radius})
        geom = UsdGeom.Sphere(prim)
        geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
        # -- target invisible
        prim = prim_utils.create_prim(f"{prim_path}/point_invisible", "Sphere", attributes={"radius": self._radius})
        geom = UsdGeom.Sphere(prim)
        geom.GetDisplayColorAttr().Set([(0.0, 0.0, 1.0)])
        prim_utils.set_prim_visibility(prim, visible=False)
        # add child reference to point instancer
        relation_manager = self._instancer_manager.GetPrototypesRel()
        relation_manager.AddTarget(f"{prim_path}/point_miss")  # target index: 0
        relation_manager.AddTarget(f"{prim_path}/point_hit")  # target index: 1
        relation_manager.AddTarget(f"{prim_path}/point_invisible")  # target index: 2

        # buffers for storing data in pixar Gf format
        # TODO: Make them very far away from the scene?
        self._proto_indices = [2] * self.count
        self._gf_positions = [Gf.Vec3f(0.0, 0.0, -10.0) for _ in range(self.count)]
        self._gf_orientations = [Gf.Quath() for _ in range(self.count)]

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
            indices: Indices to specify which alter poses. Shape is (M,), where M <= total number of markers.
                Defaults to None (i.e: all markers).
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
