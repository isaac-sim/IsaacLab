# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import carb
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import fabric as fabric_utils

logger = logging.getLogger(__name__)


class XformPrimView:
    """Optimized batched interface for reading and writing transforms of multiple USD prims.

    This class provides efficient batch operations for getting and setting poses (position and orientation)
    of multiple prims at once using torch tensors. It is designed for scenarios where you need to manipulate
    many prims simultaneously, such as in multi-agent simulations or large-scale procedural generation.

    The class supports both world-space and local-space pose operations:

    - **World poses**: Positions and orientations in the global world frame
    - **Local poses**: Positions and orientations relative to each prim's parent

    When Fabric is enabled, the class leverages NVIDIA's Fabric API for GPU-accelerated batch operations:

    - Uses `omni:fabric:worldMatrix` and `omni:fabric:localMatrix` attributes for all Boundable prims
    - Performs batch matrix decomposition/composition using Warp kernels on GPU
    - Achieves performance comparable to Isaac Sim's XFormPrim implementation
    - Works for both physics-enabled and non-physics prims (cameras, meshes, etc.).
      Note: renderers typically consume USD-authored camera transforms.

    .. warning::
        **Fabric requires CUDA**: Fabric is only supported with on CUDA devices.
        Warp's CPU backend for fabric-array writes has known issues, so attempting to use
        Fabric with CPU device (``device="cpu"``) will raise a ValueError at initialization.

    .. note::
        **Fabric Support:**

        When Fabric is enabled, this view ensures prims have the required Fabric hierarchy
        attributes (``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix``). On first Fabric
        read, USD-authored transforms initialize Fabric state. Fabric writes can optionally
        be mirrored back to USD via :attr:`sync_usd_on_fabric_write`.

        For more information, see the `Fabric Hierarchy documentation`_.

        .. _Fabric Hierarchy documentation: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/fabric_hierarchy.html

    .. note::
        **Performance Considerations:**

        * Tensor operations are performed on the specified device (CPU/CUDA)
        * USD write operations use ``Sdf.ChangeBlock`` for batched updates
        * Fabric operations use GPU-accelerated Warp kernels for maximum performance
        * For maximum performance, minimize get/set operations within tight loops

    .. note::
        **Transform Requirements:**

        All prims in the view must be Xformable and have standardized transform operations:
        ``[translate, orient, scale]``. Non-standard prims will raise a ValueError during
        initialization if :attr:`validate_xform_ops` is True. Please use the function
        :func:`isaaclab.sim.utils.standardize_xform_ops` to prepare prims before using this view.

    .. warning::
        This class operates at the USD default time code. Any animation or time-sampled data
        will not be affected by write operations. For animated transforms, you need to handle
        time-sampled keyframes separately.
    """

    def __init__(
        self,
        prim_path: str,
        device: str = "cpu",
        validate_xform_ops: bool = True,
        sync_usd_on_fabric_write: bool = False,
        stage: Usd.Stage | None = None,
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
            sync_usd_on_fabric_write: Whether to mirror Fabric transform writes back to USD.
                When True, transform updates are synchronized to USD so that USD data readers (e.g., rendering
                cameras) can observe these changes. Defaults to False for better performance.
            stage: USD stage to search for prims. Defaults to None, in which case the current active stage
                from the simulation context is used.

        Raises:
            ValueError: If any matched prim is not Xformable or doesn't have standardized
                transform operations (translate, orient, scale in that order).
        """
        # Store configuration
        self._prim_path = prim_path
        self._device = device

        # Find and validate matching prims
        stage = sim_utils.get_current_stage() if stage is None else stage
        self._prims: list[Usd.Prim] = sim_utils.find_matching_prims(prim_path, stage=stage)

        # Validate all prims have standard xform operations
        if validate_xform_ops:
            for prim in self._prims:
                sim_utils.standardize_xform_ops(prim)
                if not sim_utils.validate_standard_xform_ops(prim):
                    raise ValueError(
                        f"Prim at path '{prim.GetPath().pathString}' is not a xformable prim with standard transform"
                        f" operations [translate, orient, scale]. Received type: '{prim.GetTypeName()}'."
                        " Use sim_utils.standardize_xform_ops() to prepare the prim."
                    )

        # Determine if Fabric is supported on the device
        self._use_fabric = carb.settings.get_settings().get("/physics/fabricEnabled")
        logger.debug(f"Using Fabric for the XFormPrimView over '{self._prim_path}' on device '{self._device}'.")

        # Check for unsupported Fabric + CPU combination
        if self._use_fabric and self._device == "cpu":
            logger.warning(
                "Fabric mode with Warp fabric-array operations is not supported on CPU devices. "
                "While Fabric itself can run on both CPU and GPU, our batch Warp kernels for "
                "fabric-array operations require CUDA and are not reliable on the CPU backend. "
                "To ensure stability, Fabric is being disabled and execution will fall back "
                "to standard USD operations on the CPU. This may impact performance."
            )
            self._use_fabric = False

        # Create indices buffer
        # Since we iterate over the indices, we need to use range instead of torch tensor
        self._ALL_INDICES = list(range(len(self._prims)))

        # Some prims (e.g., Cameras) require USD-authored transforms for rendering.
        # When enabled, mirror Fabric pose writes to USD for those prims.
        self._sync_usd_on_fabric_write = sync_usd_on_fabric_write

        # Fabric batch infrastructure (initialized lazily on first use)
        self._fabric_initialized = False
        self._fabric_usd_sync_done = False
        self._fabric_selection = None
        self._fabric_to_view: wp.array | None = None
        self._view_to_fabric: wp.array | None = None
        self._default_view_indices: wp.array | None = None
        self._fabric_hierarchy = None
        # Create a valid USD attribute name: namespace:name
        # Use "isaaclab" namespace to identify our custom attributes
        self._view_index_attr = f"isaaclab:view_index:{abs(hash(self))}"

    """
    Properties.
    """

    @property
    def count(self) -> int:
        """Number of prims in this view."""
        return len(self._prims)

    @property
    def device(self) -> str:
        """Device where tensors are allocated (cpu or cuda)."""
        return self._device

    @property
    def prims(self) -> list[Usd.Prim]:
        """List of USD prims being managed by this view."""
        return self._prims

    @property
    def prim_paths(self) -> list[str]:
        """List of prim paths (as strings) for all prims being managed by this view.

        This property converts each prim to its path string representation. The conversion is
        performed lazily on first access and cached for subsequent accesses.

        Note:
            For most use cases, prefer using :attr:`prims` directly as it provides direct access
            to the USD prim objects without the conversion overhead. This property is mainly useful
            for logging, debugging, or when string paths are explicitly required.
        """
        # we cache it the first time it is accessed.
        # we don't compute it in constructor because it is expensive and we don't need it most of the time.
        # users should usually deal with prims directly as they typically need to access the prims directly.
        if not hasattr(self, "_prim_paths"):
            self._prim_paths = [prim.GetPath().pathString for prim in self._prims]
        return self._prim_paths

    """
    Operations - Setters.
    """

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set world-space poses for prims in the view.

        This method sets the position and/or orientation of each prim in world space.

        - When Fabric is enabled, the function writes directly to Fabric's ``omni:fabric:worldMatrix``
          attribute using GPU-accelerated batch operations.
        - When Fabric is disabled, the function converts to local space and writes to USD's ``xformOp:translate``
          and ``xformOp:orient`` attributes.

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
        if self._use_fabric:
            self._set_world_poses_fabric(positions, orientations, indices)
        else:
            self._set_world_poses_usd(positions, orientations, indices)

    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set local-space poses for prims in the view.

        This method sets the position and/or orientation of each prim in local space (relative to
        their parent prims).

        The function writes directly to USD's ``xformOp:translate`` and ``xformOp:orient`` attributes.

        Note:
            Even in Fabric mode, local pose operations use USD. This behavior is based on Isaac Sim's design
            where Fabric is only used for world pose operations.

            Rationale:
                - Local pose writes need correct parent-child hierarchy relationships
                - USD maintains these relationships correctly and efficiently
                - Fabric is optimized for world pose operations, not local hierarchies

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
        if self._use_fabric:
            self._set_local_poses_fabric(translations, orientations, indices)
        else:
            self._set_local_poses_usd(translations, orientations, indices)

    def set_scales(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales for prims in the view.

        This method sets the scale of each prim in the view.

        - When Fabric is enabled, the function updates scales in Fabric matrices using GPU-accelerated batch operations.
        - When Fabric is disabled, the function writes to USD's ``xformOp:scale`` attributes.

        Args:
            scales: Scales as a tensor of shape (M, 3) where M is the number of prims
                to set (either all prims if indices is None, or the number of indices provided).
            indices: Indices of prims to set scales for. Defaults to None, in which case scales are set
                for all prims in the view.

        Raises:
            ValueError: If scales shape is not (M, 3).
        """
        if self._use_fabric:
            self._set_scales_fabric(scales, indices)
        else:
            self._set_scales_usd(scales, indices)

    def set_visibility(self, visibility: torch.Tensor, indices: Sequence[int] | None = None):
        """Set visibility for prims in the view.

        This method sets the visibility of each prim in the view.

        Args:
            visibility: Visibility as a boolean tensor of shape (M,) where M is the
                number of prims to set (either all prims if indices is None, or the number of indices provided).
            indices: Indices of prims to set visibility for. Defaults to None, in which case visibility is set
                for all prims in the view.

        Raises:
            ValueError: If visibility shape is not (M,).
        """
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Validate inputs
        if visibility.shape != (len(indices_list),):
            raise ValueError(f"Expected visibility shape ({len(indices_list)},), got {visibility.shape}.")

        # Set visibility for each prim
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                # Convert prim to imageable
                imageable = UsdGeom.Imageable(self._prims[prim_idx])
                # Set visibility
                if visibility[idx]:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

    """
    Operations - Getters.
    """

    def get_world_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world-space poses for prims in the view.

        This method retrieves the position and orientation of each prim in world space by computing
        the full transform hierarchy from the prim to the world root.

        - When Fabric is enabled, the function uses Fabric batch operations with Warp kernels.
        - When Fabric is disabled, the function uses USD XformCache.

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
        if self._use_fabric:
            return self._get_world_poses_fabric(indices)
        else:
            return self._get_world_poses_usd(indices)

    def get_local_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local-space poses for prims in the view.

        This method retrieves the position and orientation of each prim in local space (relative to
        their parent prims). It reads directly from USD's ``xformOp:translate`` and ``xformOp:orient`` attributes.

        Note:
            Even in Fabric mode, local pose operations use USD. This behavior is based on Isaac Sim's design
            where Fabric is only used for world pose operations.

            Rationale:
                - Local pose reads need correct parent-child hierarchy relationships
                - USD maintains these relationships correctly and efficiently
                - Fabric is optimized for world pose operations, not local hierarchies

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
        if self._use_fabric:
            return self._get_local_poses_fabric(indices)
        else:
            return self._get_local_poses_usd(indices)

    def get_scales(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales for prims in the view.

        This method retrieves the scale of each prim in the view.

        - When Fabric is enabled, the function extracts scales from Fabric matrices using batch operations with
          Warp kernels.
        - When Fabric is disabled, the function reads from USD's ``xformOp:scale`` attributes.

        Args:
            indices: Indices of prims to get scales for. Defaults to None, in which case scales are retrieved
                for all prims in the view.

        Returns:
            A tensor of shape (M, 3) containing the scales of each prim, where M is the number of prims queried.
        """
        if self._use_fabric:
            return self._get_scales_fabric(indices)
        else:
            return self._get_scales_usd(indices)

    def get_visibility(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get visibility for prims in the view.

        This method retrieves the visibility of each prim in the view.

        Args:
            indices: Indices of prims to get visibility for. Defaults to None, in which case visibility is retrieved
                for all prims in the view.

        Returns:
            A tensor of shape (M,) containing the visibility of each prim, where M is the number of prims queried.
            The tensor is of type bool.
        """
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            # Convert to list if it is a tensor array
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        visibility = torch.zeros(len(indices_list), dtype=torch.bool, device=self._device)

        for idx, prim_idx in enumerate(indices_list):
            # Get prim
            imageable = UsdGeom.Imageable(self._prims[prim_idx])
            # Get visibility
            visibility[idx] = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible

        return visibility

    """
    Internal Functions - USD.
    """

    def _set_world_poses_usd(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set world poses to USD."""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            # Convert to list if it is a tensor array
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

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
            orientations_array = Vt.QuatdArray.FromNumpy(orientations.cpu().numpy())
        else:
            orientations_array = None

        # Create xform cache instance
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Set poses for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                # Get prim
                prim = self._prims[prim_idx]
                # Get parent prim for local space conversion
                parent_prim = prim.GetParent()

                # Determine what to set
                world_pos = positions_array[idx] if positions_array is not None else None
                world_quat = orientations_array[idx] if orientations_array is not None else None

                # Convert world pose to local if we have a valid parent
                # Note: We don't use :func:`isaaclab.sim.utils.transforms.convert_world_pose_to_local`
                #   here since it isn't optimized for batch operations.
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

    def _set_local_poses_usd(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set local poses to USD."""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Validate inputs
        if translations is not None:
            if translations.shape != (len(indices_list), 3):
                raise ValueError(f"Expected translations shape ({len(indices_list)}, 3), got {translations.shape}.")
            translations_array = Vt.Vec3dArray.FromNumpy(translations.cpu().numpy())
        else:
            translations_array = None
        if orientations is not None:
            if orientations.shape != (len(indices_list), 4):
                raise ValueError(f"Expected orientations shape ({len(indices_list)}, 4), got {orientations.shape}.")
            orientations_array = Vt.QuatdArray.FromNumpy(orientations.cpu().numpy())
        else:
            orientations_array = None

        # Set local poses
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                if translations_array is not None:
                    prim.GetAttribute("xformOp:translate").Set(translations_array[idx])
                if orientations_array is not None:
                    prim.GetAttribute("xformOp:orient").Set(orientations_array[idx])

    def _set_scales_usd(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales to USD."""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Validate inputs
        if scales.shape != (len(indices_list), 3):
            raise ValueError(f"Expected scales shape ({len(indices_list)}, 3), got {scales.shape}.")

        scales_array = Vt.Vec3dArray.FromNumpy(scales.cpu().numpy())
        # Set scales for each prim
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                prim.GetAttribute("xformOp:scale").Set(scales_array[idx])

    def _get_world_poses_usd(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world poses from USD."""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            # Convert to list if it is a tensor array
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        positions = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
        # Create xform cache instance
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Note: We don't use :func:`isaaclab.sim.utils.transforms.resolve_prim_pose`
        #   here since it isn't optimized for batch operations.
        for idx, prim_idx in enumerate(indices_list):
            # Get prim
            prim = self._prims[prim_idx]
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
        return positions, orientations  # type: ignore

    def _get_local_poses_usd(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local poses from USD."""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        translations = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))

        # Create a fresh XformCache to avoid stale cached values
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            prim_tf = xform_cache.GetLocalTransformation(prim)[0]
            prim_tf.Orthonormalize()
            translations[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        translations = torch.tensor(np.array(translations), dtype=torch.float32, device=self._device)
        orientations = torch.tensor(np.array(orientations), dtype=torch.float32, device=self._device)
        return translations, orientations  # type: ignore

    def _get_scales_usd(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales from USD."""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        scales = Vt.Vec3dArray(len(indices_list))

        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            scales[idx] = prim.GetAttribute("xformOp:scale").Get()

        # Convert to tensor
        return torch.tensor(np.array(scales), dtype=torch.float32, device=self._device)

    """
    Internal Functions - Fabric.
    """

    def _set_world_poses_fabric(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set world poses using Fabric GPU batch operations.

        Writes directly to Fabric's ``omni:fabric:worldMatrix`` attribute using Warp kernels.
        Changes are propagated through Fabric's hierarchy system but remain GPU-resident.

        For workflows mixing Fabric world pose writes with USD local pose queries, note
        that local poses read from USD's xformOp:* attributes, which may not immediately
        reflect Fabric changes. For best performance and consistency, use Fabric methods
        exclusively (get_world_poses/set_world_poses with Fabric enabled).
        """
        # Lazy initialization
        if not self._fabric_initialized:
            self._initialize_fabric()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Convert torch to warp (if provided), use dummy arrays for None to avoid Warp kernel issues
        if positions is not None:
            positions_wp = wp.from_torch(positions)
        else:
            positions_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        if orientations is not None:
            orientations_wp = wp.from_torch(orientations)
        else:
            orientations_wp = wp.zeros((0, 4), dtype=wp.float32).to(self._device)

        # Dummy array for scales (not modifying)
        scales_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        # Use cached fabricarray for world matrices
        world_matrices = self._fabric_world_matrices

        # Batch compose matrices with a single kernel launch
        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,  # dummy array instead of None
                False,  # broadcast_positions
                False,  # broadcast_orientations
                False,  # broadcast_scales
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Synchronize to ensure kernel completes
        wp.synchronize()

        # Update world transforms within Fabric hierarchy
        self._fabric_hierarchy.update_world_xforms()
        # Fabric now has authoritative data; skip future USD syncs
        self._fabric_usd_sync_done = True
        # Mirror to USD for renderer-facing prims when enabled.
        if self._sync_usd_on_fabric_write:
            self._set_world_poses_usd(positions, orientations, indices)

        # Fabric writes are GPU-resident; local pose operations still use USD.

    def _set_local_poses_fabric(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set local poses using USD (matches Isaac Sim's design).

        Note: Even in Fabric mode, local pose operations use USD.
        This is Isaac Sim's design: the ``usd=False`` parameter only affects world poses.

        Rationale:
        - Local pose writes need correct parent-child hierarchy relationships
        - USD maintains these relationships correctly and efficiently
        - Fabric is optimized for world pose operations, not local hierarchies
        """
        self._set_local_poses_usd(translations, orientations, indices)

    def _set_scales_fabric(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales using Fabric GPU batch operations."""
        # Lazy initialization
        if not self._fabric_initialized:
            self._initialize_fabric()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Convert torch to warp
        scales_wp = wp.from_torch(scales)

        # Dummy arrays for positions and orientations (not modifying)
        positions_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)
        orientations_wp = wp.zeros((0, 4), dtype=wp.float32).to(self._device)

        # Use cached fabricarray for world matrices
        world_matrices = self._fabric_world_matrices

        # Batch compose matrices on GPU with a single kernel launch
        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,  # dummy array instead of None
                orientations_wp,  # dummy array instead of None
                scales_wp,
                False,  # broadcast_positions
                False,  # broadcast_orientations
                False,  # broadcast_scales
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Synchronize to ensure kernel completes before syncing
        wp.synchronize()

        # Update world transforms to propagate changes
        self._fabric_hierarchy.update_world_xforms()
        # Fabric now has authoritative data; skip future USD syncs
        self._fabric_usd_sync_done = True
        # Mirror to USD for renderer-facing prims when enabled.
        if self._sync_usd_on_fabric_write:
            self._set_scales_usd(scales, indices)

    def _get_world_poses_fabric(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world poses from Fabric using GPU batch operations."""
        # Lazy initialization of Fabric infrastructure
        if not self._fabric_initialized:
            self._initialize_fabric()
        # Sync once from USD to ensure reads see the latest authored transforms
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Use pre-allocated buffers for full reads, allocate only for partial reads
        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            # Full read: Use cached buffers (zero allocation overhead!)
            positions_wp = self._fabric_positions_buffer
            orientations_wp = self._fabric_orientations_buffer
            scales_wp = self._fabric_dummy_buffer
        else:
            # Partial read: Need to allocate buffers of appropriate size
            positions_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32).to(self._device)
            scales_wp = self._fabric_dummy_buffer  # Always use dummy for scales

        # Use cached fabricarray for world matrices
        # This eliminates the 0.06-0.30ms variability from creating fabricarray each call
        world_matrices = self._fabric_world_matrices

        # Launch GPU kernel to decompose matrices in parallel
        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,  # dummy array instead of None
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Return tensors: zero-copy for cached buffers, conversion for partial reads
        if use_cached_buffers:
            # Zero-copy! The Warp kernel wrote directly into the PyTorch tensors
            # We just need to synchronize to ensure the kernel is done
            wp.synchronize()
            return self._fabric_positions_torch, self._fabric_orientations_torch
        else:
            # Partial read: Need to convert from Warp to torch
            positions = wp.to_torch(positions_wp)
            orientations = wp.to_torch(orientations_wp)
            return positions, orientations

    def _get_local_poses_fabric(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local poses using USD (matches Isaac Sim's design).

        Note:
            Even in Fabric mode, local pose operations use USD's XformCache.
            This is Isaac Sim's design: the ``usd=False`` parameter only affects world poses.

        Rationale:
            - Local pose computation requires parent transforms which may not be in the view
            - USD's XformCache provides efficient hierarchy-aware local transform queries
            - Fabric is optimized for world pose operations, not local hierarchies
        """
        return self._get_local_poses_usd(indices)

    def _get_scales_fabric(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales from Fabric using GPU batch operations."""
        # Lazy initialization
        if not self._fabric_initialized:
            self._initialize_fabric()
        # Sync once from USD to ensure reads see the latest authored transforms
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Use pre-allocated buffers for full reads, allocate only for partial reads
        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            # Full read: Use cached buffers (zero allocation overhead!)
            scales_wp = self._fabric_scales_buffer
        else:
            # Partial read: Need to allocate buffer of appropriate size
            scales_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)

        # Always use dummy buffers for positions and orientations (not needed for scales)
        positions_wp = self._fabric_dummy_buffer
        orientations_wp = self._fabric_dummy_buffer

        # Use cached fabricarray for world matrices
        world_matrices = self._fabric_world_matrices

        # Launch GPU kernel to decompose matrices in parallel
        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,  # dummy array instead of None
                orientations_wp,  # dummy array instead of None
                scales_wp,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Return tensor: zero-copy for cached buffers, conversion for partial reads
        if use_cached_buffers:
            # Zero-copy! The Warp kernel wrote directly into the PyTorch tensor
            wp.synchronize()
            return self._fabric_scales_torch
        else:
            # Partial read: Need to convert from Warp to torch
            return wp.to_torch(scales_wp)

    """
    Internal Functions - Initialization.
    """

    def _initialize_fabric(self) -> None:
        """Initialize Fabric batch infrastructure for GPU-accelerated pose queries.

        This method ensures all prims have the required Fabric hierarchy attributes
        (``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix``) and creates the necessary
        infrastructure for batch GPU operations using Warp.

        Based on the Fabric Hierarchy documentation, when Fabric Scene Delegate is enabled,
        all boundable prims should have these attributes. This method ensures they exist
        and are properly synchronized with USD.
        """
        import usdrt
        from usdrt import Rt

        # Get USDRT (Fabric) stage
        stage_id = sim_utils.get_current_stage_id()
        fabric_stage = usdrt.Usd.Stage.Attach(stage_id)

        # Step 1: Ensure all prims have Fabric hierarchy attributes
        # According to the documentation, these attributes are created automatically
        # when Fabric Scene Delegate is enabled, but we ensure they exist
        for i in range(self.count):
            rt_prim = fabric_stage.GetPrimAtPath(self.prim_paths[i])
            rt_xformable = Rt.Xformable(rt_prim)

            # Create Fabric hierarchy world matrix attribute if it doesn't exist
            has_attr = (
                rt_xformable.HasFabricHierarchyWorldMatrixAttr()
                if hasattr(rt_xformable, "HasFabricHierarchyWorldMatrixAttr")
                else False
            )
            if not has_attr:
                rt_xformable.CreateFabricHierarchyWorldMatrixAttr()

            # Best-effort USD->Fabric sync; authoritative initialization happens on first read.
            rt_xformable.SetWorldXformFromUsd()

            # Create view index attribute for batch operations
            rt_prim.CreateAttribute(self._view_index_attr, usdrt.Sdf.ValueTypeNames.UInt, custom=True)
            rt_prim.GetAttribute(self._view_index_attr).Set(i)

        # After syncing all prims, update the Fabric hierarchy to ensure world matrices are computed
        self._fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            fabric_stage.GetFabricId(), fabric_stage.GetStageIdAsStageId()
        )
        self._fabric_hierarchy.update_world_xforms()

        # Step 2: Create index arrays for batch operations
        self._default_view_indices = wp.zeros((self.count,), dtype=wp.uint32).to(self._device)
        wp.launch(
            kernel=fabric_utils.arange_k,
            dim=self.count,
            inputs=[self._default_view_indices],
            device=self._device,
        )
        wp.synchronize()  # Ensure indices are ready

        # Step 3: Create Fabric selection with attribute filtering
        # SelectPrims expects device format like "cuda:0" not "cuda"
        #
        # KNOWN ISSUE: SelectPrims may return prims in a different order than self._prims
        # (which comes from USD's find_matching_prims). We create a bidirectional mapping
        # (_view_to_fabric and _fabric_to_view) to handle this ordering difference.
        # This works correctly for full-view operations but partial indexing still has issues.
        fabric_device = self._device
        if self._device == "cuda":
            logger.warning("Fabric device is not specified, defaulting to 'cuda:0'.")
            fabric_device = "cuda:0"

        self._fabric_selection = fabric_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, self._view_index_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite),
            ],
            device=fabric_device,
        )

        # Step 4: Create bidirectional mapping between view and fabric indices
        self._view_to_fabric = wp.zeros((self.count,), dtype=wp.uint32).to(self._device)
        self._fabric_to_view = wp.fabricarray(self._fabric_selection, self._view_index_attr)

        wp.launch(
            kernel=fabric_utils.set_view_to_fabric_array,
            dim=self._fabric_to_view.shape[0],
            inputs=[self._fabric_to_view, self._view_to_fabric],
            device=self._device,
        )
        # Synchronize to ensure mapping is ready before any operations
        wp.synchronize()

        # Pre-allocate reusable output buffers for read operations
        self._fabric_positions_torch = torch.zeros((self.count, 3), dtype=torch.float32, device=self._device)
        self._fabric_orientations_torch = torch.zeros((self.count, 4), dtype=torch.float32, device=self._device)
        self._fabric_scales_torch = torch.zeros((self.count, 3), dtype=torch.float32, device=self._device)

        # Create Warp views of the PyTorch tensors
        self._fabric_positions_buffer = wp.from_torch(self._fabric_positions_torch, dtype=wp.float32)
        self._fabric_orientations_buffer = wp.from_torch(self._fabric_orientations_torch, dtype=wp.float32)
        self._fabric_scales_buffer = wp.from_torch(self._fabric_scales_torch, dtype=wp.float32)

        # Dummy array for unused outputs (always empty)
        self._fabric_dummy_buffer = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        # Cache fabricarray for world matrices to avoid recreation overhead
        # Refs: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usdrt_prim_selection.html
        #       https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/scenegraph_use.html
        self._fabric_world_matrices = wp.fabricarray(self._fabric_selection, "omni:fabric:worldMatrix")

        # Cache Fabric stage to avoid expensive get_current_stage() calls
        self._fabric_stage = fabric_stage

        self._fabric_initialized = True
        # Force a one-time USD->Fabric sync on first read to pick up any USD edits
        # made after the view was constructed.
        self._fabric_usd_sync_done = False

    def _sync_fabric_from_usd_once(self) -> None:
        """Sync Fabric world matrices from USD once, on the first read."""
        # Ensure Fabric is initialized
        if not self._fabric_initialized:
            self._initialize_fabric()

        # Read authoritative transforms from USD and write once into Fabric.
        positions_usd, orientations_usd = self._get_world_poses_usd()
        scales_usd = self._get_scales_usd()

        prev_sync = self._sync_usd_on_fabric_write
        self._sync_usd_on_fabric_write = False
        self._set_world_poses_fabric(positions_usd, orientations_usd)
        self._set_scales_fabric(scales_usd)
        self._sync_usd_on_fabric_write = prev_sync

        self._fabric_usd_sync_done = True

    def _resolve_indices_wp(self, indices: Sequence[int] | None) -> wp.array:
        """Resolve view indices as a Warp array."""
        if indices is None or indices == slice(None):
            if self._default_view_indices is None:
                raise RuntimeError("Fabric indices are not initialized.")
            return self._default_view_indices
        indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        return wp.array(indices_list, dtype=wp.uint32).to(self._device)
