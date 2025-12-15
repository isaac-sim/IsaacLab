# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Xform prim wrapper using USD core APIs.

This module provides a pure USD implementation of XFormPrim that doesn't depend on
Isaac Sim or Omniverse-specific APIs. It supports multiple prims (views) using regex patterns.
"""

from __future__ import annotations

import logging
import numpy as np
import re
import torch
from collections.abc import Sequence

from pxr import Gf, Usd, UsdGeom

from isaaclab.sim.utils.stage import get_current_stage

logger = logging.getLogger(__name__)


class XFormPrim:
    """Wrapper around USD Xformable prims for managing transformations.

    This class provides a simplified interface for working with one or more USD Xformable prims,
    handling transformations (translation, rotation, scale) using pure USD core APIs.
    It supports regex patterns to match multiple prims.

    Args:
        prim_paths_expr: Prim path or regex pattern to match prims. Can also be a list of paths/patterns.
            Example: "/World/Env[1-5]/Robot" will match /World/Env1/Robot, /World/Env2/Robot, etc.
        name: Optional name for this view. Defaults to "xform_prim_view".
        positions: Optional initial world positions (N, 3) or (3,).
        translations: Optional initial local translations (N, 3) or (3,).
        orientations: Optional initial orientations as quaternions (N, 4) or (4,), in (w,x,y,z) format.
        scales: Optional initial scales (N, 3) or (3,).
        reset_xform_properties: If True, resets transform properties to canonical set. Defaults to True.
        stage: The USD stage. If None, will get the current stage.

    Raises:
        ValueError: If no prims match the provided path expression.
        ValueError: If both positions and translations are specified.
    """

    def __init__(
        self,
        prim_paths_expr: str | list[str],
        name: str = "xform_prim_view",
        positions: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        translations: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        orientations: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        scales: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        reset_xform_properties: bool = True,
        stage: Usd.Stage | None = None,
    ):
        """Initialize the XFormPrim wrapper."""
        self._name = name
        self._stage = stage if stage is not None else get_current_stage()

        # Convert to list if single string
        if not isinstance(prim_paths_expr, list):
            prim_paths_expr = [prim_paths_expr]

        # Resolve regex patterns to actual prim paths
        self._prim_paths = self._resolve_prim_paths(prim_paths_expr)

        if not self._prim_paths:
            raise ValueError(f"No prims found matching patterns: {prim_paths_expr}")

        # Get all prims
        self._prims = [self._stage.GetPrimAtPath(path) for path in self._prim_paths]
        self._count = len(self._prims)

        # Validate all prims
        for i, prim in enumerate(self._prims):
            if not prim.IsValid():
                raise ValueError(f"Invalid prim at path: {self._prim_paths[i]}")

        # Reset xform properties if requested
        if reset_xform_properties:
            self._set_xform_properties()

        # Check for conflicting arguments
        if translations is not None and positions is not None:
            raise ValueError("Cannot specify both translations and positions")

        # Apply initial transformations if provided
        if positions is not None or translations is not None or orientations is not None:
            if translations is not None:
                self.set_local_poses(translations, orientations)
            else:
                self.set_world_poses(positions, orientations)

        if scales is not None:
            self.set_local_scales(scales)

    def _resolve_prim_paths(self, patterns: list[str]) -> list[str]:
        """Resolve regex patterns to actual prim paths.

        Args:
            patterns: List of prim path patterns (can include regex).

        Returns:
            List of resolved prim paths.
        """
        from isaaclab.sim.utils.prims import find_matching_prim_paths
        resolved_paths = []

        for pattern in patterns:
            # Check if pattern contains regex characters
            resolved_paths.extend(find_matching_prim_paths(pattern, self._stage))

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in resolved_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

        return unique_paths

    @property
    def count(self) -> int:
        """Get the number of prims in this view."""
        return self._count

    @property
    def prim_paths(self) -> list[str]:
        """Get list of all prim paths in this view."""
        return self._prim_paths.copy()

    @property
    def name(self) -> str:
        """Get the name of this view."""
        return self._name

    def initialize(self) -> None:
        """Initialize the prims (compatibility method).

        This method is provided for compatibility with Isaac Sim's XFormPrim interface.
        For pure USD implementation, initialization happens in __init__.
        """
        pass

    def _set_xform_properties(self) -> None:
        """Set xform properties to the canonical set: translate, orient, scale.

        This removes any non-standard rotation properties and ensures all prims
        have the standard xform operations in the correct order.
        """
        # Get current poses to restore after modifying xform ops
        current_positions, current_orientations = self.get_world_poses()

        # Properties to remove (non-standard rotation ops and transforms)
        properties_to_remove = [
            "xformOp:rotateX",
            "xformOp:rotateXZY",
            "xformOp:rotateY",
            "xformOp:rotateYXZ",
            "xformOp:rotateYZX",
            "xformOp:rotateZ",
            "xformOp:rotateZYX",
            "xformOp:rotateZXY",
            "xformOp:rotateXYZ",
            "xformOp:transform",
        ]

        for i in range(self._count):
            prim = self._prims[i]
            prop_names = prim.GetPropertyNames()
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            # Remove non-standard properties
            for prop_name in prop_names:
                if prop_name in properties_to_remove:
                    prim.RemoveProperty(prop_name)

            # Set up scale op
            if "xformOp:scale" not in prop_names:
                xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
                xform_op_scale.Set(Gf.Vec3d(1.0, 1.0, 1.0))
            else:
                xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

                # Handle unitsResolve scale factor if present
                if "xformOp:scale:unitsResolve" in prop_names:
                    current_scale = np.array(prim.GetAttribute("xformOp:scale").Get())
                    units_scale = np.array(prim.GetAttribute("xformOp:scale:unitsResolve").Get())
                    new_scale = current_scale * units_scale
                    # Convert to Python floats for USD
                    prim.GetAttribute("xformOp:scale").Set(
                        Gf.Vec3d(float(new_scale[0]), float(new_scale[1]), float(new_scale[2]))
                    )
                    prim.RemoveProperty("xformOp:scale:unitsResolve")

            # Set up translate op
            if "xformOp:translate" not in prop_names:
                xform_op_translate = xformable.AddXformOp(
                    UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
                )
            else:
                xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))

            # Set up orient (quaternion rotation) op
            if "xformOp:orient" not in prop_names:
                xform_op_rot = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
            else:
                xform_op_rot = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))

            # Set the xform op order: translate, orient, scale
            xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

        # Restore the original poses
        self.set_world_poses(positions=current_positions, orientations=current_orientations)

    def _to_numpy(self, value: torch.Tensor | np.ndarray | Sequence[float] | None) -> np.ndarray | None:
        """Convert input to numpy array."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            return value
        else:
            return np.array(value)

    def set_world_poses(
        self,
        positions: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        orientations: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> None:
        """Set world poses of the prims.

        Args:
            positions: World positions (N, 3) or (3,). If None, positions are not changed.
            orientations: World orientations as quaternions (N, 4) or (4,), in (w,x,y,z) format.
                If None, orientations are not changed.
            indices: Indices of prims to update. If None, all prims are updated.
        """
        # Convert to numpy
        pos_np = self._to_numpy(positions)
        orient_np = self._to_numpy(orientations)
        indices_np = self._to_numpy(indices)

        # Determine which prims to update
        if indices_np is None:
            prim_indices = range(self._count)
        else:
            prim_indices = indices_np.astype(int)

        # Broadcast if needed
        if pos_np is not None:
            if pos_np.ndim == 1:
                pos_np = np.tile(pos_np, (len(prim_indices), 1))

        if orient_np is not None:
            if orient_np.ndim == 1:
                orient_np = np.tile(orient_np, (len(prim_indices), 1))

        # Update each prim
        for idx, prim_idx in enumerate(prim_indices):
            prim = self._prims[prim_idx]
            xformable = UsdGeom.Xformable(prim)

            # Get or create the translate op
            translate_attr = prim.GetAttribute("xformOp:translate")
            if translate_attr:
                translate_op = UsdGeom.XformOp(translate_attr)
            else:
                translate_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)

            # Get or create the orient op
            orient_attr = prim.GetAttribute("xformOp:orient")
            if orient_attr:
                orient_op = UsdGeom.XformOp(orient_attr)
            else:
                orient_op = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble)

            # Set position
            if pos_np is not None:
                # Convert numpy values to Python floats for USD
                translate_op.Set(Gf.Vec3d(float(pos_np[idx, 0]), float(pos_np[idx, 1]), float(pos_np[idx, 2])))

            # Set orientation
            if orient_np is not None:
                # Convert numpy values to Python floats for USD
                w = float(orient_np[idx, 0])
                x = float(orient_np[idx, 1])
                y = float(orient_np[idx, 2])
                z = float(orient_np[idx, 3])
                quat = Gf.Quatd(w, Gf.Vec3d(x, y, z))
                orient_op.Set(quat)

    def set_local_poses(
        self,
        translations: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        orientations: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> None:
        """Set local poses of the prims (relative to parent).

        Args:
            translations: Local translations (N, 3) or (3,).
            orientations: Local orientations as quaternions (N, 4) or (4,), in (w,x,y,z) format.
            indices: Indices of prims to update. If None, all prims are updated.
        """
        # For local poses, we use the same method since USD xform ops are inherently local
        self.set_world_poses(positions=translations, orientations=orientations, indices=indices)

    def set_local_scales(
        self,
        scales: torch.Tensor | np.ndarray | Sequence[float],
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> None:
        """Set local scales of the prims.

        Args:
            scales: Scale factors (N, 3) or (3,).
            indices: Indices of prims to update. If None, all prims are updated.
        """
        scales_np = self._to_numpy(scales)
        indices_np = self._to_numpy(indices)

        # Determine which prims to update
        if indices_np is None:
            prim_indices = range(self._count)
        else:
            prim_indices = indices_np.astype(int)

        # Broadcast if needed
        if scales_np.ndim == 1:
            scales_np = np.tile(scales_np, (len(prim_indices), 1))

        # Update each prim
        for idx, prim_idx in enumerate(prim_indices):
            prim = self._prims[prim_idx]
            scale_attr = prim.GetAttribute("xformOp:scale")
            if scale_attr:
                # Convert numpy values to Python floats for USD
                scale_attr.Set(Gf.Vec3d(float(scales_np[idx, 0]), float(scales_np[idx, 1]), float(scales_np[idx, 2])))

    def get_world_poses(
        self,
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get world poses of the prims.

        Args:
            indices: Indices of prims to query. If None, all prims are queried.

        Returns:
            A tuple of (positions, orientations) where:
            - positions is a (N, 3) numpy array
            - orientations is a (N, 4) numpy array in (w,x,y,z) format
        """
        indices_np = self._to_numpy(indices)

        # Determine which prims to query
        if indices_np is None:
            prim_indices = range(self._count)
        else:
            prim_indices = indices_np.astype(int)

        positions = []
        orientations = []

        for prim_idx in prim_indices:
            prim = self._prims[prim_idx]
            xformable = UsdGeom.Xformable(prim)

            # Get world transform matrix
            xform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

            # Extract translation
            translation = xform_matrix.ExtractTranslation()
            positions.append([translation[0], translation[1], translation[2]])

            # Extract rotation as quaternion
            rotation = xform_matrix.ExtractRotation()
            quat = rotation.GetQuat()
            # USD uses (real, i, j, k) which is (w, x, y, z)
            orientations.append(
                [quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]]
            )

        return np.array(positions, dtype=np.float32), np.array(orientations, dtype=np.float32)

    def get_local_poses(
        self,
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get local poses of the prims (relative to parent).

        Args:
            indices: Indices of prims to query. If None, all prims are queried.

        Returns:
            A tuple of (translations, orientations) where:
            - translations is a (N, 3) numpy array
            - orientations is a (N, 4) numpy array in (w,x,y,z) format
        """
        indices_np = self._to_numpy(indices)

        # Determine which prims to query
        if indices_np is None:
            prim_indices = range(self._count)
        else:
            prim_indices = indices_np.astype(int)

        translations = []
        orientations = []

        for prim_idx in prim_indices:
            prim = self._prims[prim_idx]

            # Get local transform operations
            translate_attr = prim.GetAttribute("xformOp:translate")
            orient_attr = prim.GetAttribute("xformOp:orient")

            # Get translation
            if translate_attr:
                trans = translate_attr.Get()
                translations.append([trans[0], trans[1], trans[2]])
            else:
                translations.append([0.0, 0.0, 0.0])

            # Get orientation
            if orient_attr:
                quat = orient_attr.Get()
                # USD quaternion is (real, i, j, k) which is (w, x, y, z)
                orientations.append(
                    [quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]]
                )
            else:
                orientations.append([1.0, 0.0, 0.0, 0.0])

        return np.array(translations, dtype=np.float32), np.array(orientations, dtype=np.float32)

    def get_local_scales(
        self,
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Get local scales of the prims.

        Args:
            indices: Indices of prims to query. If None, all prims are queried.

        Returns:
            A (N, 3) numpy array of scale factors.
        """
        indices_np = self._to_numpy(indices)

        # Determine which prims to query
        if indices_np is None:
            prim_indices = range(self._count)
        else:
            prim_indices = indices_np.astype(int)

        scales = []

        for prim_idx in prim_indices:
            prim = self._prims[prim_idx]
            scale_attr = prim.GetAttribute("xformOp:scale")

            if scale_attr:
                scale = scale_attr.Get()
                scales.append([scale[0], scale[1], scale[2]])
            else:
                scales.append([1.0, 1.0, 1.0])

        return np.array(scales, dtype=np.float32)

    def set_visibilities(
        self,
        visibilities: np.ndarray | torch.Tensor | list,
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> None:
        """Set visibilities of the prims.

        Args:
            visibilities: Boolean array indicating visibility (N,).
            indices: Indices of prims to update. If None, all prims are updated.
        """
        visibilities_np = self._to_numpy(visibilities)
        indices_np = self._to_numpy(indices)

        # Determine which prims to update
        if indices_np is None:
            prim_indices = range(self._count)
        else:
            prim_indices = indices_np.astype(int)

        # Broadcast if needed
        if visibilities_np.ndim == 0 or len(visibilities_np) == 1:
            visibilities_np = np.full(len(prim_indices), visibilities_np.item())

        # Update each prim
        for idx, prim_idx in enumerate(prim_indices):
            prim = self._prims[prim_idx]
            imageable = UsdGeom.Imageable(prim)
            if visibilities_np[idx]:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def get_visibilities(
        self,
        indices: np.ndarray | list | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Get visibilities of the prims.

        Args:
            indices: Indices of prims to query. If None, all prims are queried.

        Returns:
            Boolean array indicating visibility (N,).
        """
        indices_np = self._to_numpy(indices)

        # Determine which prims to query
        if indices_np is None:
            prim_indices = range(self._count)
        else:
            prim_indices = indices_np.astype(int)

        visibilities = []

        for prim_idx in prim_indices:
            prim = self._prims[prim_idx]
            imageable = UsdGeom.Imageable(prim)
            is_visible = imageable.ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible
            visibilities.append(is_visible)

        return np.array(visibilities, dtype=bool)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"XFormPrim(name='{self._name}', count={self._count})"
