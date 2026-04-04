# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XR anchor management for IsaacTeleop-based teleoperation."""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from scipy.spatial.transform import Rotation

import carb

from .xr_anchor_utils import XrAnchorSynchronizer
from .xr_cfg import XrCfg

# Import XR components with fallback for testing
XRCore = None
XRCoreEventType = None
with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRCoreEventType

with contextlib.suppress(ModuleNotFoundError):
    from isaacsim.core.prims import SingleXFormPrim

logger = logging.getLogger(__name__)


class XrAnchorManager:
    """Manages XR anchor prim creation, synchronization, and world transform computation.

    This class is responsible for:

    1. Creating the XR anchor prim in the USD stage
    2. Configuring carb settings for XR rendering
    3. Managing the :class:`XrAnchorSynchronizer` that keeps the anchor
       aligned with a reference prim (for dynamic anchoring)
    4. Computing the 4x4 world transform matrix that converts OpenXR
       local-space poses into the Isaac Lab world frame
    """

    # Basis-change rotation from OpenXR (Y-up) to USD/Isaac Lab (Z-up).
    _OXR_TO_USD_ROTATION: np.ndarray = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, xr_cfg: XrCfg):
        """Initialize the XR anchor manager.

        Creates the anchor prim, configures carb XR settings, and sets up
        the optional anchor synchronizer for dynamic anchoring.

        Args:
            xr_cfg: XR configuration specifying anchor position, rotation,
                and optional dynamic anchoring prim path.
        """
        self._xr_cfg = xr_cfg
        self._xr_core = XRCore.get_singleton() if XRCore is not None else None
        self._xr_pre_sync_update_subscription = None

        # Resolve the headset anchor path
        if self._xr_cfg.anchor_prim_path is not None:
            anchor_path = self._xr_cfg.anchor_prim_path
            if anchor_path.endswith("/"):
                anchor_path = anchor_path[:-1]
            self._xr_anchor_headset_path = f"{anchor_path}/XRAnchor"
        else:
            self._xr_anchor_headset_path = "/World/XRAnchor"

        # Create the XR anchor prim in USD.
        # XrCfg.anchor_rot is xyzw; SingleXFormPrim expects wxyz.
        x, y, z, w = self._xr_cfg.anchor_rot
        try:
            _ = SingleXFormPrim(
                self._xr_anchor_headset_path,
                position=self._xr_cfg.anchor_pos,
                orientation=np.array([w, x, y, z], dtype=np.float64),
            )
        except Exception as e:
            logger.warning(f"Failed to create XR anchor prim: {e}")

        # Configure carb settings for XR rendering
        if hasattr(carb, "settings"):
            carb.settings.get_settings().set_float("/persistent/xr/render/nearPlane", self._xr_cfg.near_plane)
            carb.settings.get_settings().set_string("/persistent/xr/anchorMode", "custom anchor")
            carb.settings.get_settings().set_string("/xrstage/customAnchor", self._xr_anchor_headset_path)

        self._anchor_sync: XrAnchorSynchronizer | None = None
        if self._xr_core is not None:
            try:
                self._anchor_sync = XrAnchorSynchronizer(
                    xr_core=self._xr_core,
                    xr_cfg=self._xr_cfg,
                    xr_anchor_headset_path=self._xr_anchor_headset_path,
                )
                # Subscribe to pre_sync_update to keep anchor in sync each frame.
                # Capture the synchronizer in a local to satisfy type narrowing.
                if XRCoreEventType is not None:
                    assert self._anchor_sync is not None  # guaranteed by the lines above
                    anchor_sync = self._anchor_sync
                    self._xr_pre_sync_update_subscription = (
                        self._xr_core.get_message_bus().create_subscription_to_pop_by_type(
                            XRCoreEventType.pre_sync_update,
                            lambda _, _sync=anchor_sync: _sync.sync_headset_to_anchor(),
                            name="isaaclab_teleop_xr_pre_sync_update",
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize anchor synchronizer: {e}")

    @property
    def xr_core(self):
        """The XRCore singleton, or ``None`` if XR is not available."""
        return self._xr_core

    @property
    def anchor_headset_path(self) -> str:
        """The USD path of the XR anchor prim."""
        return self._xr_anchor_headset_path

    def get_world_matrix(self) -> np.ndarray:
        """Build the combined 4x4 transform from OpenXR local space to Isaac Lab world.

        This matrix performs two operations on every pose that comes out of
        IsaacTeleop's DeviceIO pipeline:

        1. **Axis conversion** -- rotates from the OpenXR coordinate convention
           (Y-up, +X right, +Z back) to the Isaac Lab convention
           (Z-up, +X forward, +Y left).
        2. **World offset** -- translates and rotates from the XR anchor's
           local frame into the Isaac Lab world frame using the anchor
           position and orientation.

        The returned matrix is ``world_T_anchor @ oxr_to_usd`` so that
        ``ControllerTransform`` can apply ``p_world = R @ p_oxr + t`` in a
        single operation.

        Strategy:
            * When the :class:`XrAnchorSynchronizer` is available (typical
              runtime), reads the cached transform that was written to the XR
              core via ``set_world_transform_matrix`` during the most recent
              ``pre_sync_update``.  This works for both dynamic anchoring
              (``anchor_prim_path`` set) and static anchoring.
            * Falls back to raw ``anchor_pos`` / ``anchor_rot`` config values
              only when the XR core is unavailable (e.g. unit tests).

        Returns:
            A (4, 4) float32 numpy array.
        """
        if self._anchor_sync is not None:
            xform = self._anchor_sync.get_world_transform()
            if xform is not None:
                pos, quat_xyzw = xform
                return self._build_matrix(pos, quat_xyzw)

        # Fallback when XR core is unavailable (e.g. unit tests).
        return self._build_matrix(self._xr_cfg.anchor_pos, self._xr_cfg.anchor_rot)

    def _build_matrix(self, pos, quat_xyzw) -> np.ndarray:
        """Assemble world_T_anchor @ oxr_to_usd as a single 4x4 matrix.

        The anchor rotation/translation places the XR origin in the Isaac Lab
        world.  The axis-conversion rotation (``_OXR_TO_USD_ROTATION``) is
        composed on the right so that it is applied *first* to the raw OpenXR
        pose before the anchor transform maps it into the world.
        """
        r_anchor = Rotation.from_quat(
            [
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
                float(quat_xyzw[3]),
            ]
        ).as_matrix()

        # Combined rotation: R_anchor @ R_oxr_to_usd
        r_combined = r_anchor @ self._OXR_TO_USD_ROTATION

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = r_combined
        mat[:3, 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
        return mat

    def reset(self) -> None:
        """Reset the anchor synchronizer state."""
        if self._anchor_sync is not None:
            self._anchor_sync.reset()

    def toggle_anchor_rotation(self) -> None:
        """Toggle anchor rotation following on the synchronizer."""
        if self._anchor_sync is not None:
            self._anchor_sync.toggle_anchor_rotation()

    def cleanup(self) -> None:
        """Release event subscriptions."""
        self._xr_pre_sync_update_subscription = None
