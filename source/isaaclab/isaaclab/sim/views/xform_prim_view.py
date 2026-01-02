# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils


class XFormPrimView:
    """Optimized batched interface for reading and writing transforms of multiple USD prims.

    This class provides efficient batch operations for getting and setting poses (position and orientation)
    of multiple prims at once using torch tensors. It is designed for scenarios where you need to manipulate
    many prims simultaneously, such as in multi-agent simulations or large-scale procedural generation.

    The class supports both world-space and local-space pose operations:

    - **World poses**: Positions and orientations in the global world frame
    - **Local poses**: Positions and orientations relative to each prim's parent

    .. note::
        **Performance Considerations:**

        * Tensor operations are performed on the specified device (CPU/CUDA)
        * USD write operations use ``Sdf.ChangeBlock`` for batched updates
        * Getting poses involves USD API calls and cannot be fully accelerated on GPU
        * For maximum performance, minimize get/set operations within tight loops

    .. note::
        **Transform Requirements:**

        All prims in the view must be Xformable and have standardized transform operations:
        ``[translate, orient, scale]``. Non-standard prims will raise a ValueError during
        initialization. Use :func:`isaaclab.sim.utils.standardize_xform_ops` to prepare prims.

    .. warning::
        This class operates at the USD default time code. Any animation or time-sampled data
        will not be affected by write operations. For animated transforms, you need to handle
        time-sampled keyframes separately.
    """

    def __init__(self, prim_path: str, device: str = "cpu", stage: Usd.Stage | None = None):
        """Initialize the XFormPrimView with matching prims.

        This method searches the USD stage for all prims matching the provided path pattern,
        validates that they are Xformable with standard transform operations, and stores
        references for efficient batch operations.

        Args:
            prim_path: USD prim path pattern to match prims. Supports wildcards (``*``) and
                regex patterns (e.g., ``"/World/Env_.*/Robot"``). See
                :func:`isaaclab.sim.utils.find_matching_prims` for pattern syntax.
            device: Device to place the tensors on. Can be ``"cpu"`` or CUDA devices like
                ``"cuda:0"``. Defaults to ``"cpu"``.
            stage: USD stage to search for prims. If None, uses the current active stage
                from the simulation context. Defaults to None.

        Raises:
            ValueError: If any matched prim is not Xformable or doesn't have standardized
                transform operations (translate, orient, scale in that order).
        """
        stage = sim_utils.get_current_stage() if stage is None else stage

        # Store configuration
        self._prim_path = prim_path
        self._device = device

        # Find and validate matching prims
        self._prims: list[Usd.Prim] = sim_utils.find_matching_prims(prim_path, stage=stage)

        # Validate all prims have standard xform operations
        for prim in self._prims:
            if not sim_utils.validate_standard_xform_ops(prim):
                raise ValueError(
                    f"Prim at path '{prim.GetPath().pathString}' is not a xformable prim with standard transform"
                    f" operations [translate, orient, scale]. Received type: '{prim.GetTypeName()}'."
                    " Use sim_utils.standardize_xform_ops() to prepare the prim."
                )

    @property
    def count(self) -> int:
        """Number of prims in this view.

        Returns:
            The number of prims being managed by this view.
        """
        return len(self._prims)

    @property
    def prim_path(self) -> str:
        """Prim path pattern used to match prims."""
        return self._prim_path

    @property
    def prims(self) -> list[Usd.Prim]:
        """List of USD prims being managed by this view."""
        return self._prims

    @property
    def device(self) -> str:
        """Device where tensors are allocated (cpu or cuda)."""
        return self._device

    """
    Operations - Setters.
    """

    def set_world_poses(self, positions: torch.Tensor | None = None, orientations: torch.Tensor | None = None) -> None:
        """Set world-space poses for all prims in the view.

        This method sets the position and/or orientation of each prim in world space. The world pose
        is computed by considering the prim's parent transforms. If a prim has a parent, this method
        will convert the world pose to the appropriate local pose before setting it.

        Note:
            This operation writes to USD at the default time code. Any animation data will not be affected.

        Args:
            positions: World-space positions as a tensor of shape (N, 3) where N is the number of prims.
                If None, positions are not modified. Defaults to None.
            orientations: World-space orientations as quaternions (w, x, y, z) with shape (N, 4).
                If None, orientations are not modified. Defaults to None.

        Raises:
            ValueError: If positions shape is not (N, 3) or orientations shape is not (N, 4).
            ValueError: If the number of poses doesn't match the number of prims in the view.
        """
        # Validate inputs
        if positions is not None:
            if positions.shape != (self.count, 3):
                raise ValueError(
                    f"Expected positions shape ({self.count}, 3), got {positions.shape}. "
                    "Number of positions must match the number of prims in the view."
                )
            positions_list = positions.tolist() if positions is not None else None
        else:
            positions_list = None
        if orientations is not None:
            if orientations.shape != (self.count, 4):
                raise ValueError(
                    f"Expected orientations shape ({self.count}, 4), got {orientations.shape}. "
                    "Number of orientations must match the number of prims in the view."
                )
            orientations_list = orientations.tolist() if orientations is not None else None
        else:
            orientations_list = None

        # Set poses for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx, prim in enumerate(self._prims):
                # Get parent prim for local space conversion
                parent_prim = prim.GetParent()

                # Determine what to set
                world_pos = tuple(positions_list[idx]) if positions_list is not None else None
                world_quat = tuple(orientations_list[idx]) if orientations_list is not None else None

                # Convert world pose to local if we have a valid parent
                if parent_prim.IsValid() and parent_prim.GetPath() != Sdf.Path.absoluteRootPath:
                    # Get current world pose if we're only setting one component
                    if world_pos is None or world_quat is None:
                        current_pos, current_quat = sim_utils.resolve_prim_pose(prim)

                        if world_pos is None:
                            world_pos = current_pos
                        if world_quat is None:
                            world_quat = current_quat

                    # Convert to local space
                    local_pos, local_quat = sim_utils.convert_world_pose_to_local(world_pos, world_quat, parent_prim)
                else:
                    # No parent or parent is root, world == local
                    local_pos = world_pos
                    local_quat = world_quat

                # Get or create the standard transform operations
                xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
                xform_op_orient = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))

                # set the data
                xform_ops = [xform_op_translate, xform_op_orient]
                xform_values = [Gf.Vec3d(*local_pos), Gf.Quatd(*local_quat)]  # type: ignore
                for xform_op, value in zip(xform_ops, xform_values):
                    if value is not None:
                        current_value = xform_op.Get()
                        xform_op.Set(type(current_value)(value) if current_value is not None else value)

    def set_local_poses(
        self, translations: torch.Tensor | None = None, orientations: torch.Tensor | None = None
    ) -> None:
        """Set local-space poses for all prims in the view.

        This method sets the position and/or orientation of each prim in local space (relative to
        their parent prims). This is useful when you want to directly manipulate the prim's transform
        attributes without considering the parent hierarchy.

        Note:
            This operation writes to USD at the default time code. Any animation data will not be affected.

        Args:
            translations: Local-space translations as a tensor of shape (N, 3) where N is the number of prims.
                If None, translations are not modified. Defaults to None.
            orientations: Local-space orientations as quaternions (w, x, y, z) with shape (N, 4).
                If None, orientations are not modified. Defaults to None.

        Raises:
            ValueError: If translations shape is not (N, 3) or orientations shape is not (N, 4).
            ValueError: If the number of poses doesn't match the number of prims in the view.
        """
        # Validate inputs
        if translations is not None:
            if translations.shape != (self.count, 3):
                raise ValueError(
                    f"Expected translations shape ({self.count}, 3), got {translations.shape}. "
                    "Number of translations must match the number of prims in the view."
                )
            translations_list = translations.tolist() if translations is not None else None
        else:
            translations_list = None
        if orientations is not None:
            if orientations.shape != (self.count, 4):
                raise ValueError(
                    f"Expected orientations shape ({self.count}, 4), got {orientations.shape}. "
                    "Number of orientations must match the number of prims in the view."
                )
            orientations_list = orientations.tolist() if orientations is not None else None
        else:
            orientations_list = None
        # Set local poses for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx, prim in enumerate(self._prims):
                local_pos = Gf.Vec3d(*translations_list[idx]) if translations_list is not None else None
                local_quat = Gf.Quatd(*orientations_list[idx]) if orientations_list is not None else None

                # Get or create the standard transform operations
                xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
                xform_op_orient = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))

                # set the data
                xform_ops = [xform_op_translate, xform_op_orient]
                xform_values = [local_pos, local_quat]
                for xform_op, value in zip(xform_ops, xform_values):
                    if value is not None:
                        current_value = xform_op.Get()
                        xform_op.Set(type(current_value)(value) if current_value is not None else value)

    def set_scales(self, scales: torch.Tensor):
        """Set scales for all prims in the view.

        This method sets the scale of each prim in the view.

        Args:
            scales: Scales as a tensor of shape (N, 3) where N is the number of prims.
        """
        # Validate inputs
        if scales.shape != (self.count, 3):
            raise ValueError(f"Expected scales shape ({self.count}, 3), got {scales.shape}.")

        scales_list = scales.tolist()
        # Set scales for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx, prim in enumerate(self._prims):
                scale = Gf.Vec3d(*scales_list[idx])
                xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))
                current_value = xform_op_scale.Get()
                xform_op_scale.Set(type(current_value)(*scale) if current_value is not None else scale)

    """
    Operations - Getters.
    """

    def get_world_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world-space poses for all prims in the view.

        This method retrieves the position and orientation of each prim in world space by computing
        the full transform hierarchy from the prim to the world root.

        Note:
            Scale and skew are ignored. The returned poses contain only translation and rotation.

        Returns:
            A tuple of (positions, orientations) where:

            - positions: Torch tensor of shape (N, 3) containing world-space positions (x, y, z)
            - orientations: Torch tensor of shape (N, 4) containing world-space quaternions (w, x, y, z)
        """
        positions = []
        orientations = []

        for prim in self._prims:
            pos, quat = sim_utils.resolve_prim_pose(prim)
            positions.append(pos)
            orientations.append(quat)

        # Convert to tensors
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self._device)
        orientations_tensor = torch.tensor(orientations, dtype=torch.float32, device=self._device)

        return positions_tensor, orientations_tensor

    def get_local_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local-space poses for all prims in the view.

        This method retrieves the position and orientation of each prim in local space (relative to
        their parent prims). These are the raw transform values stored on each prim.

        Note:
            Scale is ignored. The returned poses contain only translation and rotation.

        Returns:
            A tuple of (translations, orientations) where:

            - translations: Torch tensor of shape (N, 3) containing local-space translations (x, y, z)
            - orientations: Torch tensor of shape (N, 4) containing local-space quaternions (w, x, y, z)
        """
        translations = []
        orientations = []

        for prim in self._prims:
            local_pos, local_quat = sim_utils.resolve_prim_pose(prim, ref_prim=prim.GetParent())
            translations.append(local_pos)
            orientations.append(local_quat)

        # Convert to tensors
        translations_tensor = torch.tensor(translations, dtype=torch.float32, device=self._device)
        orientations_tensor = torch.tensor(orientations, dtype=torch.float32, device=self._device)

        return translations_tensor, orientations_tensor

    def get_scales(self) -> torch.Tensor:
        """Get scales for all prims in the view.

        This method retrieves the scale of each prim in the view.

        Returns:
            A tensor of shape (N, 3) containing the scales of each prim.
        """
        scales = []
        for prim in self._prims:
            scale = sim_utils.resolve_prim_scale(prim)
            scales.append(scale)

        # Convert to tensor
        scales_tensor = torch.tensor(scales, dtype=torch.float32, device=self._device)
        return scales_tensor
