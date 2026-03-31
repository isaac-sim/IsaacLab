# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from pxr import Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import SettingsManager

from .xform_backend import XformBackend
from .xform_fabric_backend import FabricBackend
from .xform_usd_backend import UsdBackend

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
        self._prim_path = prim_path
        self._device = device

        # Find and validate matching prims
        stage = sim_utils.get_current_stage() if stage is None else stage
        self._prims: list[Usd.Prim] = sim_utils.find_matching_prims(prim_path, stage=stage)

        if validate_xform_ops:
            for prim in self._prims:
                sim_utils.standardize_xform_ops(prim)
                if not sim_utils.validate_standard_xform_ops(prim):
                    raise ValueError(
                        f"Prim at path '{prim.GetPath().pathString}' is not a xformable prim with standard transform"
                        f" operations [translate, orient, scale]. Received type: '{prim.GetTypeName()}'."
                        " Use sim_utils.standardize_xform_ops() to prepare the prim."
                    )

        # Determine whether Fabric is available
        settings = SettingsManager.instance()
        use_fabric = bool(settings.get("/physics/fabricEnabled", False))

        if use_fabric and self._device not in ("cpu", "cuda", "cuda:0"):
            logger.warning(
                f"Fabric mode is not supported on device '{self._device}'. "
                "USDRT SelectPrims and Warp fabric arrays only support cuda:0. "
                "Falling back to standard USD operations. This may impact performance."
            )
            use_fabric = False

        # Index list used by visibility (USD-only)
        self._ALL_INDICES = list(range(len(self._prims)))

        # ---- Create backends ------------------------------------------------
        if use_fabric:
            self._backend: XformBackend = FabricBackend(self._prims, self._device)
            self._sync_backends: list[XformBackend] = (
                [UsdBackend(self._prims, self._device)] if sync_usd_on_fabric_write else []
            )
        else:
            self._backend = UsdBackend(self._prims, self._device)
            self._sync_backends = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
        if not hasattr(self, "_prim_paths"):
            self._prim_paths = [prim.GetPath().pathString for prim in self._prims]
        return self._prim_paths

    # ------------------------------------------------------------------
    # Operations – Setters
    # ------------------------------------------------------------------

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
        self._backend.set_world_poses(positions, orientations, indices)
        for sync in self._sync_backends:
            sync.set_world_poses(positions, orientations, indices)

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
        self._backend.set_local_poses(translations, orientations, indices)
        for sync in self._sync_backends:
            sync.set_local_poses(translations, orientations, indices)

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
        self._backend.set_scales(scales, indices)
        for sync in self._sync_backends:
            sync.set_scales(scales, indices)

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
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        if visibility.shape != (len(indices_list),):
            raise ValueError(f"Expected visibility shape ({len(indices_list)},), got {visibility.shape}.")

        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                imageable = UsdGeom.Imageable(self._prims[prim_idx])
                if visibility[idx]:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

    # ------------------------------------------------------------------
    # Operations – Getters
    # ------------------------------------------------------------------

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
        return self._backend.get_world_poses(indices)

    def get_local_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local-space poses for prims in the view.

        This method retrieves the position and orientation of each prim in local space
        (relative to their parent prims).

        When Fabric is enabled, reads ``omni:fabric:localMatrix`` and decomposes it
        using GPU batch operations.  Otherwise reads USD's ``xformOp:translate`` and
        ``xformOp:orient`` via an :class:`UsdGeom.XformCache`.

        Note:
            Scale is ignored. The returned poses contain only translation and rotation.

        Args:
            indices: Indices of prims to get poses for. Defaults to None, in which
                case poses are retrieved for all prims in the view.

        Returns:
            A tuple of (translations, orientations) where:

            - translations: Torch tensor of shape (M, 3) containing local-space
              translations (x, y, z), where M is the number of prims queried.
            - orientations: Torch tensor of shape (M, 4) containing local-space
              quaternions (x, y, z, w).
        """
        return self._backend.get_local_poses(indices)

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
        return self._backend.get_scales(indices)

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
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        visibility = torch.zeros(len(indices_list), dtype=torch.bool, device=self._device)

        for idx, prim_idx in enumerate(indices_list):
            imageable = UsdGeom.Imageable(self._prims[prim_idx])
            visibility[idx] = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible

        return visibility
