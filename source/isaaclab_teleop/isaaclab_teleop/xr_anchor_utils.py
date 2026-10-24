# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for synchronizing XR anchor pose with a reference prim and XR config."""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from pxr import Gf as pxrGf

from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id

from .xr_cfg import XrAnchorRotationMode

with contextlib.suppress(ModuleNotFoundError):
    import usdrt
    from usdrt import Rt


class _XrAnchorPosePolicy:
    """Own the shared reference-relative anchor policy, without Kit or Stage I/O."""

    def __init__(self, xr_cfg):
        self._xr_cfg = xr_cfg
        self.reset()

    def reset(self):
        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True
        self.__cached_world_pos = None
        self.__cached_world_quat_xyzw = None

    def toggle_anchor_rotation(self):
        self.__anchor_rotation_enabled = not self.__anchor_rotation_enabled
        logger.info("XR: Toggling anchor rotation: %s", self.__anchor_rotation_enabled)

    def get_world_transform(self):
        if self.__cached_world_pos is None:
            return None
        return self.__cached_world_pos.copy(), self.__cached_world_quat_xyzw.copy()

    def update(self, world_T_reference, dt: float | Callable[[], float], previous_headpose=None):
        if world_T_reference is not None:
            rt_matrix = pxrGf.Matrix4d(world_T_reference.T.tolist())
            rt_pos = rt_matrix.ExtractTranslation()
            if self.__anchor_prim_initial_quat is None:
                self.__anchor_prim_initial_quat = rt_matrix.ExtractRotationQuat()
            if self._xr_cfg.fixed_anchor_height:
                if self.__anchor_prim_initial_height is None:
                    self.__anchor_prim_initial_height = rt_pos[2]
                rt_pos[2] = self.__anchor_prim_initial_height
            pxr_anchor_pos = rt_pos + pxrGf.Vec3d(*self._xr_cfg.anchor_pos)
        else:
            rt_matrix = None
            pxr_anchor_pos = pxrGf.Vec3d(*self._xr_cfg.anchor_pos)
        x, y, z, w = self._xr_cfg.anchor_rot
        pxr_cfg_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))

        pxr_anchor_quat = pxr_cfg_quat

        if rt_matrix is not None:
            if self._xr_cfg.anchor_rotation_mode in (
                XrAnchorRotationMode.FOLLOW_PRIM,
                XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
            ):
                rt_prim_quat = rt_matrix.ExtractRotationQuat()
                rt_delta_quat = rt_prim_quat * self.__anchor_prim_initial_quat.GetInverse()
                pxr_delta_quat = pxrGf.Quatd(rt_delta_quat.GetReal(), pxrGf.Vec3d(*rt_delta_quat.GetImaginary()))

                # yaw-only about Z (right-handed, Z-up)
                wq = pxr_delta_quat.GetReal()
                ix, iy, iz = pxr_delta_quat.GetImaginary()
                yaw = math.atan2(2.0 * (wq * iz + ix * iy), 1.0 - 2.0 * (iy * iy + iz * iz))
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                pxr_delta_yaw_only_quat = pxrGf.Quatd(cy, pxrGf.Vec3d(0.0, 0.0, sy))
                pxr_anchor_quat = pxr_delta_yaw_only_quat * pxr_cfg_quat

                if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
                    if self.__smoothed_anchor_quat is None:
                        self.__smoothed_anchor_quat = pxr_anchor_quat
                    else:
                        step_dt = dt() if callable(dt) else dt
                        alpha = 1.0 - math.exp(-step_dt / max(self._xr_cfg.anchor_rotation_smoothing_time, 1e-6))
                        alpha = min(1.0, max(0.05, alpha))
                        self.__smoothed_anchor_quat = pxrGf.Slerp(alpha, self.__smoothed_anchor_quat, pxr_anchor_quat)
                        pxr_anchor_quat = self.__smoothed_anchor_quat

            elif self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.CUSTOM:
                if self._xr_cfg.anchor_rotation_custom_func is not None:
                    rt_prim_quat = rt_matrix.ExtractRotationQuat()
                    anchor_prim_pose = np.array(
                        [
                            rt_pos[0],
                            rt_pos[1],
                            rt_pos[2],
                            rt_prim_quat.GetImaginary()[0],
                            rt_prim_quat.GetImaginary()[1],
                            rt_prim_quat.GetImaginary()[2],
                            rt_prim_quat.GetReal(),
                        ],
                        dtype=np.float64,
                    )
                    prev_head = previous_headpose if previous_headpose is not None else np.zeros(7, dtype=np.float64)
                    np_array_quat = self._xr_cfg.anchor_rotation_custom_func(prev_head, anchor_prim_pose)
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

        self.__cached_world_pos = np.array([pxr_anchor_pos[0], pxr_anchor_pos[1], pxr_anchor_pos[2]], dtype=np.float64)
        fq_img = pxr_final_quat.GetImaginary()
        self.__cached_world_quat_xyzw = np.array(
            [fq_img[0], fq_img[1], fq_img[2], pxr_final_quat.GetReal()], dtype=np.float64
        )
        return pxr_mat


class XrAnchorSynchronizer:
    """Adapt the shared anchor policy to Kit Fabric and XRCore pre-sync events."""

    def __init__(self, xr_core: Any, xr_cfg: Any, xr_anchor_headset_path: str):
        self._xr_core = xr_core
        self._xr_cfg = xr_cfg
        self._xr_anchor_headset_path = xr_anchor_headset_path
        self._policy = _XrAnchorPosePolicy(xr_cfg)
        stage = get_current_stage()
        anchor = stage.GetPrimAtPath(xr_anchor_headset_path)
        self._layer_identifier = anchor.GetPrimStack()[0].layer.identifier
        self._reference_world_matrix = None
        if xr_cfg.anchor_prim_path is not None:
            rt_stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
            reference = rt_stage.GetPrimAtPath(xr_cfg.anchor_prim_path)
            if not reference.IsValid():
                raise ValueError(f"No XR reference frame at {xr_cfg.anchor_prim_path!r}.")
            self._reference_world_matrix = Rt.Xformable(reference).GetFabricHierarchyWorldMatrixAttr()

    def reset(self):
        self._policy.reset()
        self.sync_headset_to_anchor()

    def toggle_anchor_rotation(self):
        self._policy.toggle_anchor_rotation()

    def get_world_transform(self):
        return self._policy.get_world_transform()

    def sync_headset_to_anchor(self):
        """Read the Fabric reference and update the Kit anchor once per pre-sync."""
        try:
            reference = None
            if self._reference_world_matrix is not None:
                matrix = self._reference_world_matrix.Get()
                reference = np.array([[matrix[i][j] for j in range(4)] for i in range(4)]).T
            matrix = self._policy.update(
                reference,
                lambda: SimulationContext.instance().get_rendering_dt(),
                getattr(self, "_previous_headpose", None),
            )
            self._xr_core.set_world_transform_matrix(self._xr_anchor_headset_path, matrix, self._layer_identifier)
        except Exception as exc:
            logger.warning("XR: Anchor sync failed: %s", exc)
