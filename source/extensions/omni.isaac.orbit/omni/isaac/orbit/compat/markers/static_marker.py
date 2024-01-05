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
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.prims import GeometryPrim
from pxr import Gf, UsdGeom

import omni.isaac.orbit.compat.utils.kit as kit_utils
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path


class StaticMarker:
    """A class to coordinate groups of visual markers (loaded from USD).

    This class allows visualization of different UI elements in the scene, such as points and frames.
    The class uses `UsdGeom.PointInstancer`_ for efficient handling of the element in the stage
    via instancing of the marker.

    Usage:
        To create 24 default frame markers with a scale of 0.5:

        .. code-block:: python

            from omni.isaac.orbit.compat.markers import StaticMarker

            # create a static marker
            marker = StaticMarker("/World/Visuals/frames", 24, scale=(0.5, 0.5, 0.5))

            # set position of the marker
            marker_positions = np.random.uniform(-1.0, 1.0, (24, 3))
            marker.set_world_poses(marker_positions)

    .. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html

    """

    def __init__(
        self,
        prim_path: str,
        count: int,
        usd_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color: tuple[float, float, float] | None = None,
    ):
        """Initialize the class.

        When the class is initialized, the :class:`UsdGeom.PointInstancer` is created into the stage
        and the marker prim is registered into it.

        Args:
            prim_path: The prim path where the PointInstancer will be created.
            count: The number of marker duplicates to create.
            usd_path: The USD file path to the marker. Defaults to the USD path for the RGB frame axis marker.
            scale: The scale of the marker. Defaults to (1.0, 1.0, 1.0).
            color: The color of the marker. If provided, it overrides the existing color on all the
                prims of the marker. Defaults to None.

        Raises:
            ValueError: When a prim already exists at the :obj:`prim_path` and it is not a
                :class:`UsdGeom.PointInstancer`.
            FileNotFoundError: When the USD file at :obj:`usd_path` does not exist.
        """
        # resolve default markers in the UI elements
        # -- USD path
        if usd_path is None:
            usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
        else:
            if not check_file_path(usd_path):
                raise FileNotFoundError(f"USD file for the marker not found at: {usd_path}")
        # -- prim path
        stage = stage_utils.get_current_stage()
        if prim_utils.is_prim_path_valid(prim_path):
            # retrieve prim if it exists
            prim = prim_utils.get_prim_at_path(prim_path)
            if not prim.IsA(UsdGeom.PointInstancer):
                raise ValueError(f"The prim at path {prim_path} cannot be parsed as a `PointInstancer` object")
            self._instancer_manager = UsdGeom.PointInstancer(prim)
        else:
            # create a new prim
            self._instancer_manager = UsdGeom.PointInstancer.Define(stage, prim_path)
        # store inputs
        self.prim_path = prim_path
        self.count = count
        self._usd_path = usd_path

        # create manager for handling instancing of frame markers
        self._instancer_manager = UsdGeom.PointInstancer.Define(stage, prim_path)
        # create a child prim for the marker
        prim_utils.create_prim(f"{prim_path}/marker", usd_path=self._usd_path)
        # disable any physics on the marker
        # FIXME: Also support disabling rigid body properties on the marker.
        #   Currently, it is not possible on GPU pipeline.
        # kit_utils.set_nested_rigid_body_properties(f"{prim_path}/marker", rigid_body_enabled=False)
        kit_utils.set_nested_collision_properties(f"{prim_path}/marker", collision_enabled=False)
        # apply material to marker
        if color is not None:
            prim = GeometryPrim(f"{prim_path}/marker")
            material = PreviewSurface(f"{prim_path}/markerColor", color=np.asarray(color))
            prim.apply_visual_material(material, weaker_than_descendants=False)

        # add child reference to point instancer
        # FUTURE: Add support for multiple markers in the same instance manager?
        relation_manager = self._instancer_manager.GetPrototypesRel()
        relation_manager.AddTarget(f"{prim_path}/marker")  # target index: 0

        # buffers for storing data in pixar Gf format
        # FUTURE: Make them very far away from the scene?
        self._gf_positions = [Gf.Vec3f() for _ in range(self.count)]
        self._gf_orientations = [Gf.Quath() for _ in range(self.count)]
        self._gf_scales = [Gf.Vec3f(*tuple(scale)) for _ in range(self.count)]

        # specify that all vis prims are related to same geometry
        self._instancer_manager.GetProtoIndicesAttr().Set([0] * self.count)
        # set initial positions of the targets
        self._instancer_manager.GetScalesAttr().Set(self._gf_scales)
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
            positions: Positions in the world frame. Shape is (M, 3). Defaults to None, which means left unchanged.
            orientations: Quaternion orientations (w, x, y, z) in the world frame of the prims. Shape is (M, 4).
                Defaults to None, which means left unchanged.
            indices: Indices to specify which alter poses. Shape is (M,) where M <= total number of markers.
                Defaults to None (i.e: all markers).
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

    def set_scales(self, scales: np.ndarray | torch.Tensor, indices: Sequence[int] | None = None):
        """Update marker poses in the simulation world frame.

        Args:
            scales: Scale applied before any rotation is applied. Shape is (M, 3).
            indices: Indices to specify which alter poses.
                Shape is (M,), where M <= total number of markers. Defaults to None (i.e: all markers).
        """
        # default arguments
        if indices is None:
            indices = range(self.count)
        # resolve inputs
        scales = scales.tolist()
        # change marker locations
        for i, marker_index in enumerate(indices):
            self._gf_scales[marker_index][:] = scales[i]
        # apply to instance manager
        self._instancer_manager.GetScalesAttr().Set(self._gf_scales)
