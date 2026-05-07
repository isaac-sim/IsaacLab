# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import numpy as np
import torch
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import ProxyArray

from .base_frame_view import BaseFrameView

logger = logging.getLogger(__name__)


class UsdFrameView(BaseFrameView):
    """Batched interface for reading and writing transforms of multiple USD prims.

    Provides batch operations for getting and setting poses (position and orientation)
    of multiple prims at once via USD's ``XformCache``.

    The class supports both world-space and local-space pose operations:

    - **World poses**: Positions and orientations in the global world frame
    - **Local poses**: Positions and orientations relative to each prim's parent

    For GPU-accelerated Fabric operations, use the PhysX backend variant
    obtained via :class:`~isaaclab.sim.views.FrameView`.

    Pose getters return :class:`~isaaclab.utils.warp.ProxyArray`.  Setters accept ``wp.array``.

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
            device: Device to place arrays on. Can be ``"cpu"`` or CUDA devices like
                ``"cuda:0"``. Defaults to ``"cpu"``.
            validate_xform_ops: Whether to validate that the prims have standard xform operations.
                Defaults to True.
            stage: USD stage to search for prims. Defaults to None, in which case the current
                active stage from the simulation context is used.
            **kwargs: Additional keyword arguments (ignored). Allows forward-compatible
                construction when callers pass backend-specific options.

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of prims in this view."""
        return len(self._prims)

    @property
    def device(self) -> str:
        """Device where arrays are allocated (cpu or cuda)."""
        return self._device

    @property
    def prims(self) -> list[Usd.Prim]:
        """List of USD prims being managed by this view."""
        return self._prims

    @property
    def prim_paths(self) -> list[str]:
        """List of prim paths (as strings) for all prims being managed by this view.

        The conversion is performed lazily on first access and cached.
        """
        if not hasattr(self, "_prim_paths"):
            self._prim_paths = [prim.GetPath().pathString for prim in self._prims]
        return self._prim_paths

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_world_poses(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ):
        """Set world-space poses for prims in the view.

        Converts the desired world pose to local-space relative to each prim's
        parent before writing to USD xform ops.

        Args:
            positions: World-space positions of shape ``(M, 3)``.
            orientations: World-space quaternions ``(w, x, y, z)`` of shape ``(M, 4)``.
            indices: Indices of prims to set poses for. Defaults to None (all prims).
        """
        indices_list = self._resolve_indices(indices)

        positions_array = Vt.Vec3dArray.FromNumpy(self._to_numpy(positions)) if positions is not None else None
        orientations_array = Vt.QuatdArray.FromNumpy(self._to_numpy(orientations)) if orientations is not None else None

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
                    # Root-level prim: world == local
                    local_pos = world_pos
                    local_quat = world_quat

                if local_pos is not None:
                    prim.GetAttribute("xformOp:translate").Set(local_pos)
                if local_quat is not None:
                    prim.GetAttribute("xformOp:orient").Set(local_quat)

    def set_local_poses(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ):
        """Set local-space poses for prims in the view.

        Args:
            translations: Local-space translations of shape ``(M, 3)``.
            orientations: Local-space quaternions ``(w, x, y, z)`` of shape ``(M, 4)``.
            indices: Indices of prims to set poses for. Defaults to None (all prims).
        """
        indices_list = self._resolve_indices(indices)

        translations_array = Vt.Vec3dArray.FromNumpy(self._to_numpy(translations)) if translations is not None else None
        orientations_array = Vt.QuatdArray.FromNumpy(self._to_numpy(orientations)) if orientations is not None else None

        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                if translations_array is not None:
                    prim.GetAttribute("xformOp:translate").Set(translations_array[idx])
                if orientations_array is not None:
                    prim.GetAttribute("xformOp:orient").Set(orientations_array[idx])

    def set_scales(self, scales: wp.array, indices: wp.array | None = None):
        """Set scales for prims in the view.

        Args:
            scales: Scales of shape ``(M, 3)``.
            indices: Indices of prims to set scales for. Defaults to None (all prims).
        """
        indices_list = self._resolve_indices(indices)
        scales_array = Vt.Vec3dArray.FromNumpy(self._to_numpy(scales))

        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                prim.GetAttribute("xformOp:scale").Set(scales_array[idx])

    def set_visibility(self, visibility: torch.Tensor, indices: wp.array | None = None):
        """Set visibility for prims in the view.

        Args:
            visibility: Visibility as a boolean tensor of shape ``(M,)``.
            indices: Indices of prims to set visibility for. Defaults to None (all prims).
        """
        indices_list = self._resolve_indices(indices)

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
    # Getters
    # ------------------------------------------------------------------

    def get_world_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space poses for prims in the view.

        Args:
            indices: Indices of prims to get poses for. Defaults to None (all prims).

        Returns:
            A tuple ``(positions, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers. Use ``.warp`` for the underlying ``wp.array`` or ``.torch`` for a
            cached zero-copy ``torch.Tensor`` view.
        """
        indices_list = self._resolve_indices(indices)

        positions = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            prim_tf = xform_cache.GetLocalToWorldTransform(prim)
            prim_tf.Orthonormalize()
            positions[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        pos_wp = wp.array(np.array(positions, dtype=np.float32), dtype=wp.float32, device=self._device)
        quat_wp = wp.array(np.array(orientations, dtype=np.float32), dtype=wp.float32, device=self._device)
        return ProxyArray(pos_wp), ProxyArray(quat_wp)

    def get_local_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get local-space poses for prims in the view.

        Args:
            indices: Indices of prims to get poses for. Defaults to None (all prims).

        Returns:
            A tuple ``(translations, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers. Use ``.warp`` for the underlying ``wp.array`` or ``.torch`` for a
            cached zero-copy ``torch.Tensor`` view.
        """
        indices_list = self._resolve_indices(indices)

        translations = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            prim_tf = xform_cache.GetLocalTransformation(prim)[0]
            prim_tf.Orthonormalize()
            translations[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        pos_wp = wp.array(np.array(translations, dtype=np.float32), dtype=wp.float32, device=self._device)
        quat_wp = wp.array(np.array(orientations, dtype=np.float32), dtype=wp.float32, device=self._device)
        return ProxyArray(pos_wp), ProxyArray(quat_wp)

    def get_scales(self, indices: wp.array | None = None) -> wp.array:
        """Get scales for prims in the view.

        Args:
            indices: Indices of prims to get scales for. Defaults to None (all prims).

        Returns:
            A ``wp.array`` of shape ``(M, 3)``.
        """
        indices_list = self._resolve_indices(indices)

        scales = Vt.Vec3dArray(len(indices_list))
        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            scales[idx] = prim.GetAttribute("xformOp:scale").Get()

        return wp.array(np.array(scales, dtype=np.float32), dtype=wp.float32, device=self._device)

    def get_visibility(self, indices: wp.array | None = None) -> torch.Tensor:
        """Get visibility for prims in the view.

        Args:
            indices: Indices of prims to get visibility for. Defaults to None (all prims).

        Returns:
            A tensor of shape ``(M,)`` containing the visibility of each prim (bool).
        """
        indices_list = self._resolve_indices(indices)

        visibility = torch.zeros(len(indices_list), dtype=torch.bool, device=self._device)
        for idx, prim_idx in enumerate(indices_list):
            imageable = UsdGeom.Imageable(self._prims[prim_idx])
            visibility[idx] = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
        return visibility

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_indices(self, indices: wp.array | None):
        """Resolve warp indices to an iterable of ints for per-prim USD operations."""
        if indices is None or indices == slice(None):
            return self._ALL_INDICES
        return indices.numpy()

    @staticmethod
    def _to_numpy(data: wp.array | torch.Tensor) -> np.ndarray:
        """Convert a ``wp.array`` or ``torch.Tensor`` to a numpy array on CPU."""
        if isinstance(data, wp.array):
            return data.numpy()
        return data.cpu().numpy()
