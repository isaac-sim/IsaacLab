# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for synchronizing XR anchor pose with a reference prim and XR config."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from pxr import Gf as pxrGf

from isaaclab.sim import FrameView, SimulationContext
from isaaclab.sim.utils.stage import get_current_stage

from .xr_cfg import XrAnchorRotationMode

logger = logging.getLogger(__name__)


class XrAnchorSynchronizer:
    """Keeps the XR anchor prim aligned with a reference prim according to XR config."""

    def __init__(self, xr_core: Any, xr_cfg: Any, xr_anchor_headset_path: str):
        self._xr_core = xr_core
        self._xr_cfg = xr_cfg
        self._xr_anchor_headset_path = xr_anchor_headset_path

        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True
        self._previous_headpose = np.zeros(7, dtype=np.float64)

        # Cached anchor world transform (pos, quat_xyzw) set by sync_headset_to_anchor().
        # Reading back from the prim hierarchy is unreliable when the anchor is a child of a
        # physics-driven prim (e.g. the pelvis) because Fabric computes the hierarchy world
        # matrix using the physics-updated parent while the local xform was decomposed against
        # the USD-level parent, which can diverge.
        self.__cached_world_pos: np.ndarray | None = None
        self.__cached_world_quat_xyzw: np.ndarray | None = None

        stage = get_current_stage()
        anchor = stage.GetPrimAtPath(xr_anchor_headset_path)
        self.__anchor_headset_layer_identifier = anchor.GetPrimStack()[0].layer.identifier
        self._reference_frame = None
        if xr_cfg.anchor_prim_path is not None:
            self._reference_frame = FrameView(
                xr_cfg.anchor_prim_path, device=SimulationContext.instance().device, validate_xform_ops=False
            )
            if self._reference_frame.count != 1:
                self._reference_frame.close()
                self._reference_frame = None
                raise ValueError(f"Expected one XR reference frame at {xr_cfg.anchor_prim_path!r}.")
            self._reference_frame.get_world_poses()

    def close(self) -> None:
        """Release the reference frame when XR synchronization stops."""
        if self._reference_frame is not None:
            self._reference_frame.close()
            self._reference_frame = None

    def reset(self):
        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True
        self.__cached_world_pos = None
        self.__cached_world_quat_xyzw = None
        self.sync_headset_to_anchor()

    def toggle_anchor_rotation(self):
        self.__anchor_rotation_enabled = not self.__anchor_rotation_enabled
        logger.info(f"XR: Toggling anchor rotation: {self.__anchor_rotation_enabled}")

    def get_world_transform(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the anchor world transform.

        Returns the cached world transform that was computed by the most recent
        call to :meth:`sync_headset_to_anchor`.  Using the cached value avoids
        a Fabric/USD layer mismatch: when the XR anchor prim is a child of a
        physics-driven prim (e.g. the robot pelvis), reading
        ``GetFabricHierarchyWorldMatrixAttr`` would compose the Fabric-side
        parent transform (updated by physics) with a local xform that was
        decomposed against the USD-side parent (which can lag behind),
        producing an incorrect world matrix that drifts as the robot moves.

        Returns:
            A ``(position, quat_xyzw)`` tuple of numpy float64 arrays,
            or ``None`` if :meth:`sync_headset_to_anchor` has not run yet.
        """
        if self.__cached_world_pos is not None and self.__cached_world_quat_xyzw is not None:
            return self.__cached_world_pos, self.__cached_world_quat_xyzw
        return None

    def sync_headset_to_anchor(self):
        """Sync XR anchor pose in USD for both dynamic and static anchoring.

        For **dynamic** anchoring (``anchor_prim_path`` is set), the reference
        prim's world position is read through its frame view and ``anchor_pos`` is added
        as an offset.  For **static** anchoring (no prim path), ``anchor_pos``
        is used directly as the world position.

        In both cases the function calls ``set_world_transform_matrix`` on the
        XR core so that the rendering anchor and the pipeline's
        ``world_T_anchor`` matrix are guaranteed to agree, and caches the
        world transform for :meth:`get_world_transform`.
        """
        try:
            if self._reference_frame is not None:
                position, orientation = self._reference_frame.get_world_poses()
                reference_pos = position.warp.numpy()[0].astype(np.float64)
                x, y, z, w = orientation.warp.numpy()[0].astype(np.float64)
                reference_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))

                if self.__anchor_prim_initial_quat is None:
                    self.__anchor_prim_initial_quat = reference_quat

                if self._xr_cfg.fixed_anchor_height:
                    if self.__anchor_prim_initial_height is None:
                        self.__anchor_prim_initial_height = reference_pos[2]
                    reference_pos[2] = self.__anchor_prim_initial_height

                pxr_anchor_pos = pxrGf.Vec3d(*reference_pos) + pxrGf.Vec3d(*self._xr_cfg.anchor_pos)
            else:
                pxr_anchor_pos = pxrGf.Vec3d(*self._xr_cfg.anchor_pos)

            x, y, z, w = self._xr_cfg.anchor_rot
            pxr_cfg_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))

            pxr_anchor_quat = pxr_cfg_quat

            if self._reference_frame is not None:
                if self._xr_cfg.anchor_rotation_mode in (
                    XrAnchorRotationMode.FOLLOW_PRIM,
                    XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
                ):
                    delta_quat = reference_quat * self.__anchor_prim_initial_quat.GetInverse()

                    # yaw-only about Z (right-handed, Z-up)
                    wq = delta_quat.GetReal()
                    ix, iy, iz = delta_quat.GetImaginary()
                    yaw = math.atan2(2.0 * (wq * iz + ix * iy), 1.0 - 2.0 * (iy * iy + iz * iz))
                    cy = math.cos(yaw * 0.5)
                    sy = math.sin(yaw * 0.5)
                    pxr_delta_yaw_only_quat = pxrGf.Quatd(cy, pxrGf.Vec3d(0.0, 0.0, sy))
                    pxr_anchor_quat = pxr_delta_yaw_only_quat * pxr_cfg_quat

                    if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
                        if self.__smoothed_anchor_quat is None:
                            self.__smoothed_anchor_quat = pxr_anchor_quat
                        else:
                            dt = SimulationContext.instance().get_rendering_dt()
                            alpha = 1.0 - math.exp(-dt / max(self._xr_cfg.anchor_rotation_smoothing_time, 1e-6))
                            alpha = min(1.0, max(0.05, alpha))
                            self.__smoothed_anchor_quat = pxrGf.Slerp(
                                alpha, self.__smoothed_anchor_quat, pxr_anchor_quat
                            )
                            pxr_anchor_quat = self.__smoothed_anchor_quat

                elif self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.CUSTOM:
                    if self._xr_cfg.anchor_rotation_custom_func is not None:
                        anchor_prim_pose = np.array(
                            [*reference_pos, *reference_quat.GetImaginary(), reference_quat.GetReal()], dtype=np.float64
                        )
                        np_array_quat = self._xr_cfg.anchor_rotation_custom_func(
                            self._previous_headpose, anchor_prim_pose
                        )
                        x, y, z, w = np_array_quat
                        pxr_anchor_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))

            pxr_mat = pxrGf.Matrix4d()
            pxr_mat.SetTranslateOnly(pxr_anchor_pos)

            if self.__anchor_rotation_enabled:
                pxr_final_quat = pxr_anchor_quat
                self.__last_anchor_quat = pxr_anchor_quat
            else:
                if self.__last_anchor_quat is None:
                    self.__last_anchor_quat = pxr_anchor_quat
                pxr_final_quat = self.__last_anchor_quat
                self.__smoothed_anchor_quat = self.__last_anchor_quat

            pxr_mat.SetRotateOnly(pxr_final_quat)

            self.__cached_world_pos = np.array(
                [pxr_anchor_pos[0], pxr_anchor_pos[1], pxr_anchor_pos[2]], dtype=np.float64
            )
            fq_img = pxr_final_quat.GetImaginary()
            self.__cached_world_quat_xyzw = np.array(
                [fq_img[0], fq_img[1], fq_img[2], pxr_final_quat.GetReal()], dtype=np.float64
            )

            self._xr_core.set_world_transform_matrix(
                self._xr_anchor_headset_path, pxr_mat, self.__anchor_headset_layer_identifier
            )
        except Exception as e:
            logger.warning(f"XR: Anchor sync failed: {e}")
