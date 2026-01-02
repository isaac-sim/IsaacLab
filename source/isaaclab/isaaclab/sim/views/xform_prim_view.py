# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils


class XformPrimView:
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

    def __init__(
        self, prim_path: str, device: str = "cpu", validate_xform_ops: bool = True, stage: Usd.Stage | None = None
    ):
        """Initialize the view with matching prims.

        This method searches the USD stage for all prims matching the provided path pattern,
        validates that they are Xformable with standard transform operations, and stores
        references for efficient batch operations.

        We generally recommend to validate the xform operations, as it ensures that the prims are in a consistent state
        and have the standard transform operations (translate, orient, scale in that order).
        However, if you are sure that the prims are in a consistent state, you can set this to False to improve
        performance. This can save around 45-50% of the time taken to initialize the view.

        Args:
            prim_path: USD prim path pattern to match prims. Supports wildcards (``*``) and
                regex patterns (e.g., ``"/World/Env_.*/Robot"``). See
                :func:`isaaclab.sim.utils.find_matching_prims` for pattern syntax.
            device: Device to place the tensors on. Can be ``"cpu"`` or CUDA devices like
                ``"cuda:0"``. Defaults to ``"cpu"``.
            validate_xform_ops: Whether to validate that the prims have standard xform operations.
                Defaults to True.
            stage: USD stage to search for prims. Defaults to None, in which case the current active stage
                from the simulation context is used.

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

        # Create indices buffer
        # Since we iterate over the indices, we need to use range instead of torch tensor
        self._ALL_INDICES = list(range(len(self._prims)))

        # Validate all prims have standard xform operations
        if validate_xform_ops:
            for prim in self._prims:
                if not sim_utils.validate_standard_xform_ops(prim):
                    raise ValueError(
                        f"Prim at path '{prim.GetPath().pathString}' is not a xformable prim with standard transform"
                        f" operations [translate, orient, scale]. Received type: '{prim.GetTypeName()}'."
                        " Use sim_utils.standardize_xform_ops() to prepare the prim."
                    )

    """
    Properties.
    """

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

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ):
        """Set world-space poses for prims in the view.

        This method sets the position and/or orientation of each prim in world space. The world pose
        is computed by considering the prim's parent transforms. If a prim has a parent, this method
        will convert the world pose to the appropriate local pose before setting it.

        Note:
            This operation writes to USD at the default time code. Any animation data will not be affected.

        Args:
            positions: World-space positions as a tensor of shape (M, 3) where M is the number of prims
                to set (either all prims if indices is None, or the number of indices provided).
                Defaults to None, in which case positions are not modified.
            orientations: World-space orientations as quaternions (w, x, y, z) with shape (M, 4).
                Defaults to None, in which case orientations are not modified.
            indices: Indices of prims to set poses for. Defaults to None, in which case poses are set
                for all prims in the view.

        Raises:
            ValueError: If positions shape is not (M, 3) or orientations shape is not (M, 4).
            ValueError: If the number of poses doesn't match the number of indices provided.
        """
        # Resolve indices
        indices_list = self._ALL_INDICES if indices is None else indices.tolist()

        # Validate inputs
        if positions is not None:
            if positions.shape != (len(indices_list), 3):
                raise ValueError(
                    f"Expected positions shape ({len(indices_list)}, 3), got {positions.shape}. "
                    "Number of positions must match the number of prims in the view."
                )
            positions_array = Vt.Vec3dArray.FromNumpy(positions.cpu().numpy())
        else:
            positions_array = None
        if orientations is not None:
            if orientations.shape != (len(indices_list), 4):
                raise ValueError(
                    f"Expected orientations shape ({len(indices_list)}, 4), got {orientations.shape}. "
                    "Number of orientations must match the number of prims in the view."
                )
            # Vt expects quaternions in xyzw order
            orientations_array = Vt.QuatdArray.FromNumpy(math_utils.convert_quat(orientations, to="xyzw").cpu().numpy())
        else:
            orientations_array = None

        # Create xform cache instance
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Set poses for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx in indices_list:
                # Get prim
                prim = self._prims[idx]
                # Get parent prim for local space conversion
                parent_prim = prim.GetParent()

                # Determine what to set
                world_pos = positions_array[idx] if positions_array is not None else None
                world_quat = orientations_array[idx] if orientations_array is not None else None

                # Convert world pose to local if we have a valid parent
                if parent_prim.IsValid() and parent_prim.GetPath() != Sdf.Path.absoluteRootPath:
                    # Get current world pose if we're only setting one component
                    if positions_array is None or orientations_array is None:
                        # get prim xform
                        prim_tf = xform_cache.GetLocalToWorldTransform(prim)
                        # sanitize quaternion
                        # this is needed, otherwise the quaternion might be non-normalized
                        prim_tf.Orthonormalize()
                        # populate desired world transform
                        if world_pos is not None:
                            prim_tf.SetTranslateOnly(world_pos)
                        if world_quat is not None:
                            prim_tf.SetRotateOnly(world_quat)
                    else:
                        # Both position and orientation are provided, create new transform
                        prim_tf = Gf.Matrix4d()
                        prim_tf.SetTranslateOnly(world_pos)
                        prim_tf.SetRotateOnly(world_quat)

                    # Convert to local space
                    parent_world_tf = xform_cache.GetLocalToWorldTransform(parent_prim)
                    local_tf = prim_tf * parent_world_tf.GetInverse()
                    local_pos = local_tf.ExtractTranslation()
                    local_quat = local_tf.ExtractRotationQuat()
                else:
                    # No parent or parent is root, world == local
                    local_pos = world_pos
                    local_quat = world_quat

                # Get or create the standard transform operations
                if local_pos is not None:
                    prim.GetAttribute("xformOp:translate").Set(local_pos)
                if local_quat is not None:
                    prim.GetAttribute("xformOp:orient").Set(local_quat)

    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Set local-space poses for prims in the view.

        This method sets the position and/or orientation of each prim in local space (relative to
        their parent prims). This is useful when you want to directly manipulate the prim's transform
        attributes without considering the parent hierarchy.

        Note:
            This operation writes to USD at the default time code. Any animation data will not be affected.

        Args:
            translations: Local-space translations as a tensor of shape (M, 3) where M is the number of prims
                to set (either all prims if indices is None, or the number of indices provided).
                Defaults to None, in which case translations are not modified.
            orientations: Local-space orientations as quaternions (w, x, y, z) with shape (M, 4).
                Defaults to None, in which case orientations are not modified.
            indices: Indices of prims to set poses for. Defaults to None, in which case poses are set
                for all prims in the view.

        Raises:
            ValueError: If translations shape is not (M, 3) or orientations shape is not (M, 4).
            ValueError: If the number of poses doesn't match the number of indices provided.
        """
        # Resolve indices
        indices_list = self._ALL_INDICES if indices is None else indices.tolist()

        # Validate inputs
        if translations is not None:
            if translations.shape != (len(indices_list), 3):
                raise ValueError(
                    f"Expected translations shape ({len(indices_list)}, 3), got {translations.shape}. "
                    "Number of translations must match the number of prims in the view."
                )
            translations_array = Vt.Vec3dArray.FromNumpy(translations.cpu().numpy())
        else:
            translations_array = None
        if orientations is not None:
            if orientations.shape != (len(indices_list), 4):
                raise ValueError(
                    f"Expected orientations shape ({len(indices_list)}, 4), got {orientations.shape}. "
                    "Number of orientations must match the number of prims in the view."
                )
            # Vt expects quaternions in xyzw order
            orientations_array = Vt.QuatdArray.FromNumpy(math_utils.convert_quat(orientations, to="xyzw").cpu().numpy())
        else:
            orientations_array = None
        # Set local poses for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx in indices_list:
                # Get prim
                prim = self._prims[idx]
                # Set attributes if provided
                if translations_array is not None:
                    prim.GetAttribute("xformOp:translate").Set(translations_array[idx])
                if orientations_array is not None:
                    prim.GetAttribute("xformOp:orient").Set(orientations_array[idx])

    def set_scales(self, scales: torch.Tensor, indices: torch.Tensor | None = None):
        """Set scales for prims in the view.

        This method sets the scale of each prim in the view.

        Args:
            scales: Scales as a tensor of shape (M, 3) where M is the number of prims
                to set (either all prims if indices is None, or the number of indices provided).
            indices: Indices of prims to set scales for. Defaults to None, in which case scales are set
                for all prims in the view.

        Raises:
            ValueError: If scales shape is not (M, 3).
        """
        # Resolve indices
        indices_list = self._ALL_INDICES if indices is None else indices.tolist()

        # Validate inputs
        if scales.shape != (len(indices_list), 3):
            raise ValueError(f"Expected scales shape ({len(indices_list)}, 3), got {scales.shape}.")

        scales_array = Vt.Vec3dArray.FromNumpy(scales.cpu().numpy())
        # Set scales for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx in indices_list:
                # Get prim
                prim = self._prims[idx]
                # Set scale attribute
                prim.GetAttribute("xformOp:scale").Set(scales_array[idx])

    """
    Operations - Getters.
    """

    def get_world_poses(self, indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world-space poses for prims in the view.

        This method retrieves the position and orientation of each prim in world space by computing
        the full transform hierarchy from the prim to the world root.

        Note:
            Scale and skew are ignored. The returned poses contain only translation and rotation.

        Args:
            indices: Indices of prims to get poses for. Defaults to None, in which case poses are retrieved
                for all prims in the view.

        Returns:
            A tuple of (positions, orientations) where:

            - positions: Torch tensor of shape (M, 3) containing world-space positions (x, y, z),
              where M is the number of prims queried.
            - orientations: Torch tensor of shape (M, 4) containing world-space quaternions (w, x, y, z)
        """
        # Resolve indices
        indices_list = self._ALL_INDICES if indices is None else indices.tolist()
        # Create buffers
        positions = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
        # Create xform cache instance
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for idx in indices_list:
            # Get prim
            prim = self._prims[idx]
            # get prim xform
            prim_tf = xform_cache.GetLocalToWorldTransform(prim)
            # sanitize quaternion
            # this is needed, otherwise the quaternion might be non-normalized
            prim_tf.Orthonormalize()
            # extract position and orientation
            positions[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        # move to torch tensors
        positions = torch.tensor(np.array(positions), dtype=torch.float32, device=self._device)
        orientations = torch.tensor(np.array(orientations), dtype=torch.float32, device=self._device)
        # underlying data is in xyzw order, convert to wxyz order
        orientations = math_utils.convert_quat(orientations, to="wxyz")

        return positions, orientations  # type: ignore

    def get_local_poses(self, indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local-space poses for prims in the view.

        This method retrieves the position and orientation of each prim in local space (relative to
        their parent prims). These are the raw transform values stored on each prim.

        Note:
            Scale is ignored. The returned poses contain only translation and rotation.

        Args:
            indices: Indices of prims to get poses for. Defaults to None, in which case poses are retrieved
                for all prims in the view.

        Returns:
            A tuple of (translations, orientations) where:

            - translations: Torch tensor of shape (M, 3) containing local-space translations (x, y, z),
              where M is the number of prims queried.
            - orientations: Torch tensor of shape (M, 4) containing local-space quaternions (w, x, y, z)
        """
        # Resolve indices
        indices_list = self._ALL_INDICES if indices is None else indices.tolist()
        # Create buffers
        translations = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
        # Create xform cache instance
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for idx in indices_list:
            # Get prim
            prim = self._prims[idx]
            # get prim xform
            prim_tf = xform_cache.GetLocalTransformation(prim)[0]
            # sanitize quaternion
            # this is needed, otherwise the quaternion might be non-normalized
            prim_tf.Orthonormalize()
            # extract position and orientation
            translations[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        # move to torch tensors
        translations = torch.tensor(np.array(translations), dtype=torch.float32, device=self._device)
        orientations = torch.tensor(np.array(orientations), dtype=torch.float32, device=self._device)
        # underlying data is in xyzw order, convert to wxyz order
        orientations = math_utils.convert_quat(orientations, to="wxyz")

        return translations, orientations  # type: ignore

    def get_scales(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Get scales for prims in the view.

        This method retrieves the scale of each prim in the view.

        Args:
            indices: Indices of prims to get scales for. Defaults to None, in which case scales are retrieved
                for all prims in the view.

        Returns:
            A tensor of shape (M, 3) containing the scales of each prim, where M is the number of prims queried.
        """
        # Resolve indices
        indices_list = self._ALL_INDICES if indices is None else indices.tolist()
        # Create buffers
        scales = Vt.Vec3dArray(len(indices_list))

        for idx in indices_list:
            # Get prim
            prim = self._prims[idx]
            scales[idx] = prim.GetAttribute("xformOp:scale").Get()

        # Convert to tensor
        scales = torch.tensor(np.array(scales), dtype=torch.float32, device=self._device)
        return scales
