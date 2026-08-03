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
from .xform_space_writer import FrameViewLocalSpaceWriter, FrameViewWorldSpaceWriter

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

    All writes go through the writer-scope API (:meth:`xform_world_space_writer`
    / :meth:`xform_local_space_writer`).  The
    USD backend's writers are pass-throughs: each :meth:`set_poses` /
    :meth:`set_scales` call directly modifies the prim's USD ``xformOp:*``
    attributes (no batching, no derivation step on exit) -- USD has no
    separate world-matrix storage to keep in sync.

    Getters return :class:`~isaaclab.utils.warp.ProxyArray`.

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
    # Writer factory hooks (pass-through writers; USD has no derived state)
    # ------------------------------------------------------------------

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        return _UsdWorldSpaceWriter(self)

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        return _UsdLocalSpaceWriter(self)

    # ------------------------------------------------------------------
    # Visibility (USD-only, no writer scope)
    # ------------------------------------------------------------------

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
    # Backend hooks: pose / scale writes (called by writers).
    # ------------------------------------------------------------------

    def _apply_world_pose_write(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Apply a world-space pose write directly to USD xform ops.

        Converts the desired world pose to local-space relative to each prim's
        parent before writing.
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

    def _apply_local_pose_write(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Apply a local-space pose write directly to USD xform ops."""
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

    def _apply_local_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Apply a local-space scale write (``xformOp:scale``)."""
        indices_list = self._resolve_indices(indices)
        scales_array = Vt.Vec3dArray.FromNumpy(self._to_numpy(scales))

        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                prim.GetAttribute("xformOp:scale").Set(scales_array[idx])

    def _apply_world_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Apply a world-space scale write.

        Computes ``local_scale = world_scale / parent_world_scale`` and writes
        to ``xformOp:scale``.
        """
        indices_list = self._resolve_indices(indices)
        scales_np = self._to_numpy(scales)
        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                parent = prim.GetParent()
                if parent and parent.IsValid() and parent.GetPath() != Sdf.Path.absoluteRootPath:
                    parent_world = xf_cache.GetLocalToWorldTransform(parent)
                    parent_scale = Gf.Vec3d(
                        Gf.Vec3d(parent_world[0][0], parent_world[0][1], parent_world[0][2]).GetLength(),
                        Gf.Vec3d(parent_world[1][0], parent_world[1][1], parent_world[1][2]).GetLength(),
                        Gf.Vec3d(parent_world[2][0], parent_world[2][1], parent_world[2][2]).GetLength(),
                    )
                else:
                    parent_scale = Gf.Vec3d(1.0, 1.0, 1.0)
                local_scale = Gf.Vec3d(
                    float(scales_np[idx][0] / parent_scale[0]),
                    float(scales_np[idx][1] / parent_scale[1]),
                    float(scales_np[idx][2] / parent_scale[2]),
                )
                prim.GetAttribute("xformOp:scale").Set(local_scale)

    # ------------------------------------------------------------------
    # Backend hooks: pose / scale reads.
    # ------------------------------------------------------------------

    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
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

    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
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

    def _get_local_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        indices_list = self._resolve_indices(indices)

        scales = Vt.Vec3dArray(len(indices_list))
        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            scales[idx] = prim.GetAttribute("xformOp:scale").Get()

        return ProxyArray(wp.array(np.array(scales, dtype=np.float32), dtype=wp.float32, device=self._device))

    def _get_world_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        indices_list = self._resolve_indices(indices)
        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        scales = np.empty((len(indices_list), 3), dtype=np.float32)
        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            world_mtx = xf_cache.GetLocalToWorldTransform(prim)
            scales[idx, 0] = Gf.Vec3d(world_mtx[0][0], world_mtx[0][1], world_mtx[0][2]).GetLength()
            scales[idx, 1] = Gf.Vec3d(world_mtx[1][0], world_mtx[1][1], world_mtx[1][2]).GetLength()
            scales[idx, 2] = Gf.Vec3d(world_mtx[2][0], world_mtx[2][1], world_mtx[2][2]).GetLength()

        return ProxyArray(wp.array(scales, dtype=wp.float32, device=self._device))

    # ------------------------------------------------------------------
    # Deprecated get_scales / set_scales hooks
    # ------------------------------------------------------------------

    def _get_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """USD legacy: get_scales returns local scales."""
        return self._get_local_scales_impl(indices)

    def _set_scales_impl(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """USD legacy: set_scales writes local scales via a one-shot writer scope."""
        with self.xform_local_space_writer() as writer:
            writer.set_scales(scales, indices)

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


# ----------------------------------------------------------------------
# Pass-through writer classes
# ----------------------------------------------------------------------


class _UsdWorldSpaceWriter(FrameViewWorldSpaceWriter):
    """USD world-space writer: pass-through to backend ``_apply_*`` hooks.

    USD has no separate world-matrix storage to keep in sync; ``__exit__``
    is a no-op beyond releasing the single-writer lock.
    """

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_world_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_world_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _UsdLocalSpaceWriter(FrameViewLocalSpaceWriter):
    """USD local-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_local_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_local_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)  # type: ignore[attr-defined]
