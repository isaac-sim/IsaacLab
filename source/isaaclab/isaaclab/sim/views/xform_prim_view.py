# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils

from .base_xform_prim_view import BaseXformPrimView

logger = logging.getLogger(__name__)


class XformPrimView(BaseXformPrimView):
    """Batched interface for reading and writing transforms of multiple USD prims.

    This class provides batch operations for getting and setting poses (position and orientation)
    of multiple prims at once using torch tensors via USD's ``XformCache``.

    The class supports both world-space and local-space pose operations:

    - **World poses**: Positions and orientations in the global world frame
    - **Local poses**: Positions and orientations relative to each prim's parent

    For GPU-accelerated Fabric operations, use the PhysX backend variant
    obtained via :class:`~isaaclab.sim.views.XformPrimViewFactory`.

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
        stage: Usd.Stage | None = None,
        **kwargs,
    ):
        """Initialize the view with matching prims.

        Args:
            prim_path: USD prim path pattern to match prims. Supports wildcards (``*``) and
                regex patterns (e.g., ``"/World/Env_.*/Robot"``). See
                :func:`isaaclab.sim.utils.find_matching_prims` for pattern syntax.
            device: Device to place the tensors on. Can be ``"cpu"`` or CUDA devices like
                ``"cuda:0"``. Defaults to ``"cpu"``.
            validate_xform_ops: Whether to validate that the prims have standard xform operations.
                Defaults to True.
            stage: USD stage to search for prims. Defaults to None, in which case the current
                active stage from the simulation context is used.
            **kwargs: Additional keyword arguments (ignored). Allows forward-compatible
                construction when callers pass backend-specific options like
                ``sync_usd_on_fabric_write``.

        Raises:
            ValueError: If any matched prim is not Xformable or doesn't have standardized
                transform operations (translate, orient, scale in that order).
        """
        self._prim_path = prim_path
        self._device = device

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

        self._ALL_INDICES = list(range(len(self._prims)))

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
        """
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

        Args:
            positions: World-space positions as a tensor of shape (M, 3).
                Defaults to None, in which case positions are not modified.
            orientations: World-space orientations as quaternions (w, x, y, z) with shape (M, 4).
                Defaults to None, in which case orientations are not modified.
            indices: Indices of prims to set poses for. Defaults to None (all prims).
        """
        self._set_world_poses_usd(positions, orientations, indices)

    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set local-space poses for prims in the view.

        Args:
            translations: Local-space translations as a tensor of shape (M, 3).
                Defaults to None, in which case translations are not modified.
            orientations: Local-space orientations as quaternions (w, x, y, z) with shape (M, 4).
                Defaults to None, in which case orientations are not modified.
            indices: Indices of prims to set poses for. Defaults to None (all prims).
        """
        self._set_local_poses_usd(translations, orientations, indices)

    def set_scales(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales for prims in the view.

        Args:
            scales: Scales as a tensor of shape (M, 3).
            indices: Indices of prims to set scales for. Defaults to None (all prims).
        """
        self._set_scales_usd(scales, indices)

    def set_visibility(self, visibility: torch.Tensor, indices: Sequence[int] | None = None):
        """Set visibility for prims in the view.

        Args:
            visibility: Visibility as a boolean tensor of shape (M,).
            indices: Indices of prims to set visibility for. Defaults to None (all prims).
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

    """
    Operations - Getters.
    """

    def get_world_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world-space poses for prims in the view.

        Args:
            indices: Indices of prims to get poses for. Defaults to None (all prims).

        Returns:
            A tuple of (positions, orientations) where:

            - positions: Torch tensor of shape (M, 3) containing world-space positions (x, y, z).
            - orientations: Torch tensor of shape (M, 4) containing world-space quaternions (w, x, y, z).
        """
        return self._get_world_poses_usd(indices)

    def get_local_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local-space poses for prims in the view.

        Args:
            indices: Indices of prims to get poses for. Defaults to None (all prims).

        Returns:
            A tuple of (translations, orientations) where:

            - translations: Torch tensor of shape (M, 3) containing local-space translations (x, y, z).
            - orientations: Torch tensor of shape (M, 4) containing local-space quaternions (w, x, y, z).
        """
        return self._get_local_poses_usd(indices)

    def get_scales(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales for prims in the view.

        Args:
            indices: Indices of prims to get scales for. Defaults to None (all prims).

        Returns:
            A tensor of shape (M, 3) containing the scales of each prim.
        """
        return self._get_scales_usd(indices)

    def get_visibility(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get visibility for prims in the view.

        Args:
            indices: Indices of prims to get visibility for. Defaults to None (all prims).

        Returns:
            A tensor of shape (M,) containing the visibility of each prim (bool).
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
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

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
            orientations_array = Vt.QuatdArray.FromNumpy(orientations.cpu().numpy())
        else:
            orientations_array = None

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                parent_prim = prim.GetParent()

                world_pos = positions_array[idx] if positions_array is not None else None
                world_quat = orientations_array[idx] if orientations_array is not None else None

                if parent_prim.IsValid() and parent_prim.GetPath() != Sdf.Path.absoluteRootPath:
                    if positions_array is None or orientations_array is None:
                        prim_tf = xform_cache.GetLocalToWorldTransform(prim)
                        prim_tf.Orthonormalize()
                        if world_pos is not None:
                            prim_tf.SetTranslateOnly(world_pos)
                        if world_quat is not None:
                            prim_tf.SetRotateOnly(world_quat)
                    else:
                        prim_tf = Gf.Matrix4d()
                        prim_tf.SetTranslateOnly(world_pos)
                        prim_tf.SetRotateOnly(world_quat)

                    parent_world_tf = xform_cache.GetLocalToWorldTransform(parent_prim)
                    local_tf = prim_tf * parent_world_tf.GetInverse()
                    local_pos = local_tf.ExtractTranslation()
                    local_quat = local_tf.ExtractRotationQuat()
                else:
                    local_pos = world_pos
                    local_quat = world_quat

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
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

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

        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                if translations_array is not None:
                    prim.GetAttribute("xformOp:translate").Set(translations_array[idx])
                if orientations_array is not None:
                    prim.GetAttribute("xformOp:orient").Set(orientations_array[idx])

    def _set_scales_usd(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales to USD."""
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        if scales.shape != (len(indices_list), 3):
            raise ValueError(f"Expected scales shape ({len(indices_list)}, 3), got {scales.shape}.")

        scales_array = Vt.Vec3dArray.FromNumpy(scales.cpu().numpy())
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                prim.GetAttribute("xformOp:scale").Set(scales_array[idx])

    def _get_world_poses_usd(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world poses from USD."""
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        positions = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            prim_tf = xform_cache.GetLocalToWorldTransform(prim)
            prim_tf.Orthonormalize()
            positions[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        positions = torch.tensor(np.array(positions), dtype=torch.float32, device=self._device)
        orientations = torch.tensor(np.array(orientations), dtype=torch.float32, device=self._device)
        return positions, orientations  # type: ignore

    def _get_local_poses_usd(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local poses from USD."""
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        translations = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
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
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        scales = Vt.Vec3dArray(len(indices_list))
        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            scales[idx] = prim.GetAttribute("xformOp:scale").Get()

        return torch.tensor(np.array(scales), dtype=torch.float32, device=self._device)
