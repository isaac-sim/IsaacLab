# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XR anchor management for Isaac Capture-based teleoperation."""

from __future__ import annotations

import contextlib
import logging
import threading

import numpy as np
from scipy.spatial.transform import Rotation

from .xr_anchor_utils import XrAnchorSynchronizer, _XrAnchorPosePolicy
from .xr_cfg import XrCfg

# Import Kit components with fallback for sessions without Kit
XRCore = None
XRCoreEventType = None
with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRCoreEventType

from isaaclab.sim.utils.prims import create_prim as _create_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sim.utils.transforms import convert_world_pose_to_local, standardize_xform_ops

logger = logging.getLogger(__name__)


def _create_or_update_xr_anchor(
    prim_path: str, position: tuple[float, float, float], orientation: tuple[float, float, float, float]
) -> None:
    """Apply the configured world pose to a stage-scoped XR anchor."""
    if not _xr_anchor_prim_exists(prim_path):
        _create_prim(
            prim_path,
            prim_type="Xform",
            position=np.asarray(position, dtype=np.float64),
            orientation=np.asarray(orientation, dtype=np.float64),
        )
        return
    prim = get_current_stage().GetPrimAtPath(prim_path)
    translation, rotation = convert_world_pose_to_local(tuple(position), tuple(orientation), prim.GetParent())
    if not standardize_xform_ops(prim, translation, rotation):
        raise ValueError(f"XR anchor {prim_path!r} is not transformable")


def _xr_anchor_prim_exists(prim_path: str) -> bool:
    """Return True when ``prim_path`` is already a valid prim on the active stage.

    Used to avoid re-creating the anchor prim on every
    :class:`XrAnchorManager` construction in multi-replay batches (the
    replay agent rebuilds the device per run, but the prim is
    stage-scoped). Best-effort: returns ``False`` if the stage cannot
    be inspected (e.g. before stage creation) so the caller falls
    through to the create-and-warn path used historically.
    """
    try:
        stage = get_current_stage()
        if stage is None:
            return False
        return stage.GetPrimAtPath(prim_path).IsValid()
    except Exception as exc:
        logger.debug("_xr_anchor_prim_exists(%r) failed: %s", prim_path, exc)
        return False


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

    def __init__(self, xr_cfg: XrCfg, *, ovxr_session: bool = False):
        """Initialize the XR anchor manager.

        Creates or refreshes the stage-scoped anchor's configured world pose,
        configures carb XR settings, and sets up the optional anchor
        synchronizer for dynamic anchoring. Reusing the Stage across sessions
        does not retain a previous session's anchor configuration.

        Args:
            xr_cfg: XR configuration specifying anchor position, rotation,
                and optional dynamic anchoring prim path.
            ovxr_session: An application-owned renderer/Stage supplies XR.
                This path must not configure or synchronize Kit XRCore.

        Raises:
            ValueError: An existing anchor prim is not transformable. Other anchor
                authoring errors also propagate for OVXR presentation, so
                rendering cannot continue with a pose that differs from tracking.
        """
        self._xr_cfg = xr_cfg
        self._xr_core = XRCore.get_singleton() if XRCore is not None and not ovxr_session else None
        self._xr_pre_sync_update_subscription = None
        self._external_lock = threading.RLock()
        self._external_policy = (
            _XrAnchorPosePolicy(xr_cfg) if ovxr_session and xr_cfg.anchor_prim_path is not None else None
        )
        self._external_epoch = 0
        self._external_world_matrix = None
        self._external_error = None
        self._external_render_local = None

        # Resolve the headset anchor path
        if self._xr_cfg.anchor_prim_path is not None:
            anchor_path = self._xr_cfg.anchor_prim_path
            if anchor_path.endswith("/"):
                anchor_path = anchor_path[:-1]
            self._xr_anchor_headset_path = f"{anchor_path}/XRAnchor"
        else:
            self._xr_anchor_headset_path = "/World/XRAnchor"

        render_anchor_overridden = (
            self._xr_cfg.render_anchor_pos is not None or self._xr_cfg.render_anchor_rot is not None
        )
        if render_anchor_overridden:
            if self._xr_cfg.anchor_prim_path is not None:
                anchor_path = self._xr_cfg.anchor_prim_path.rstrip("/")
                self._xr_render_anchor_path = f"{anchor_path}/XRRenderAnchor"
            else:
                self._xr_render_anchor_path = "/World/XRRenderAnchor"
        else:
            self._xr_render_anchor_path = self._xr_anchor_headset_path

        # Anchors survive per-run device teardown. Reuse their paths while
        # applying this run's configuration, rather than retaining a stale pose.
        try:
            _create_or_update_xr_anchor(self._xr_anchor_headset_path, self._xr_cfg.anchor_pos, self._xr_cfg.anchor_rot)
        except Exception as e:
            if ovxr_session:
                raise
            logger.warning(f"Failed to create or update XR anchor prim: {e}")

        if render_anchor_overridden:
            render_pos = (
                self._xr_cfg.anchor_pos if self._xr_cfg.render_anchor_pos is None else self._xr_cfg.render_anchor_pos
            )
            render_rot = (
                self._xr_cfg.anchor_rot if self._xr_cfg.render_anchor_rot is None else self._xr_cfg.render_anchor_rot
            )
            try:
                _create_or_update_xr_anchor(self._xr_render_anchor_path, render_pos, render_rot)
            except Exception as e:
                if ovxr_session:
                    raise
                logger.warning(f"Failed to create or update XR render anchor prim: {e}")

        if self._external_policy is not None and render_anchor_overridden:
            from pxr import UsdGeom

            prim = get_current_stage().GetPrimAtPath(self._xr_render_anchor_path)
            self._external_render_local = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T.copy()

        # Configure settings only when Kit owns a live XRCore. Kitless external-
        # present sessions retain this anchor but have no carb settings interface.
        if self._xr_core is not None:
            import carb.settings  # noqa: PLC0415

            settings = carb.settings.get_settings()
            settings.set_float("/persistent/xr/render/nearPlane", self._xr_cfg.near_plane)
            settings.set_string("/persistent/xr/anchorMode", "custom anchor")
            settings.set_string("/xrstage/customAnchor", self._xr_render_anchor_path)

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

    @property
    def render_anchor_path(self) -> str:
        """The USD path consumed by XR rendering."""
        return self._xr_render_anchor_path

    def get_world_matrix(self) -> np.ndarray:
        """Build the combined 4x4 transform from OpenXR local space to Isaac Lab world.

        This matrix performs two operations on every pose that comes out of
        Isaac Capture's DeviceIO pipeline:

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
            * External dynamic sessions read the cached pose sealed by their
              frame owner, using application-published physics references.
              Unseeded or failed reset epochs raise instead of returning a static pose.
            * Static and standalone sessions without Kit retain configured poses.

        Returns:
            A (4, 4) float32 numpy array.
        """
        if self._external_policy is not None:
            with self._external_lock:
                if self._external_error is not None:
                    raise RuntimeError(
                        "Dynamic XR anchor publication failed; recreate the session"
                    ) from self._external_error
                if self._external_world_matrix is None:
                    raise RuntimeError("Dynamic XR anchor requires a committed post-reset scene pose")
                matrix = self._external_world_matrix.copy()
                matrix[:3, :3] = matrix[:3, :3] @ self._OXR_TO_USD_ROTATION
                return matrix.astype(np.float32)
        if self._anchor_sync is not None:
            xform = self._anchor_sync.get_world_transform()
            if xform is not None:
                pos, quat_xyzw = xform
                return self._build_matrix(pos, quat_xyzw)

        # Static and standalone sessions retain their configured-pose fallback.
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
        if self._external_policy is not None:
            with self._external_lock:
                self._external_epoch += 1
                self._external_policy.reset()
                self._external_world_matrix = None
        if self._anchor_sync is not None:
            self._anchor_sync.reset()

    def toggle_anchor_rotation(self) -> None:
        """Toggle anchor rotation following on the synchronizer."""
        if self._external_policy is not None:
            with self._external_lock:
                self._external_policy.toggle_anchor_rotation()
        if self._anchor_sync is not None:
            self._anchor_sync.toggle_anchor_rotation()

    @property
    def external_reference_path(self) -> str | None:
        """The exact physics reference path, when external dynamic anchoring is active."""
        return self._xr_cfg.anchor_prim_path if self._external_policy is not None else None

    @property
    def external_epoch(self) -> tuple[int, bool]:
        """Return reset identity and whether the current epoch needs its first publication."""
        with self._external_lock:
            if self._external_error is not None:
                raise RuntimeError(
                    "Dynamic XR anchor publication failed; recreate the session"
                ) from self._external_error
            return self._external_epoch, self._external_world_matrix is None

    @property
    def external_anchor_paths(self) -> tuple[str, ...]:
        """Native Xforms written together by the OVXR presentation host."""
        return tuple(dict.fromkeys((self._xr_anchor_headset_path, self._xr_render_anchor_path)))

    def commit_external_pose(self, reference, rendering_dt: float, epoch: int, commit) -> bool:
        """Advance the shared policy once and expose its input pose only after native seal.

        The host serializes this operation with scene publication. Old committed
        references cannot reseed a new reset epoch. A failed write invalidates the
        cached input pose and propagates; it never falls back to configured static data.
        """
        with self._external_lock:
            if self._external_error is not None:
                raise RuntimeError(
                    "Dynamic XR anchor publication failed; recreate the session"
                ) from self._external_error
            if epoch != self._external_epoch:
                return False
            try:
                matrix = np.asarray(self._external_policy.update(reference, rendering_dt)).T.copy()
                matrices = [matrix]
                if self._external_render_local is not None:
                    matrices.append(reference @ self._external_render_local)
                commit(np.stack(matrices))
            except BaseException as exc:
                self._external_world_matrix = None
                self._external_error = exc
                raise
            self._external_world_matrix = matrix
            return True

    def cleanup(self) -> None:
        """Release event subscriptions."""
        self._xr_pre_sync_update_subscription = None
