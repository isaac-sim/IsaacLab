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
carb = None
with contextlib.suppress(ModuleNotFoundError):
    import carb

XRCore = None
XRCoreEventType = None
with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRCoreEventType

from isaaclab.sim.utils.prims import create_prim as _create_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sim.utils.transforms import convert_world_pose_to_local, standardize_xform_ops
from isaaclab.utils.version import has_kit

logger = logging.getLogger(__name__)


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

        # Create the XR anchor prim in USD if it does not already exist.
        # The check matters for multi-replay batches (the replay agent
        # rebuilds ``IsaacTeleopDevice`` -- and therefore this manager --
        # for each run, but the prim is stage-scoped and survives
        # per-run device teardown). XrCfg.anchor_rot is xyzw; create_prim
        # orientation expects xyzw.
        if not _xr_anchor_prim_exists(self._xr_anchor_headset_path):
            x, y, z, w = self._xr_cfg.anchor_rot
            try:
                pos = np.asarray(self._xr_cfg.anchor_pos, dtype=np.float64)
                quat_xyzw = np.asarray([x, y, z, w], dtype=np.float64)
                _create_prim(self._xr_anchor_headset_path, prim_type="Xform", position=pos, orientation=quat_xyzw)
            except Exception as e:
                logger.warning(f"Failed to create XR anchor prim: {e}")

        # Configure carb settings for XR rendering; carb can import while Kit is not running
        if carb is not None and has_kit():
            settings = carb.settings.get_settings()
            settings.set_float("/persistent/xr/render/nearPlane", self._xr_cfg.near_plane)
            # The ar/vr profile-specific settings default to 0.15 and take precedence
            # over the generic path above, so profile-based sessions (e.g. CloudXR)
            # would keep culling geometry closer than 15 cm unless they are written too.
            settings.set_float("/persistent/xr/profile/ar/render/nearPlane", self._xr_cfg.near_plane)
            settings.set_float("/persistent/xr/profile/vr/render/nearPlane", self._xr_cfg.near_plane)
            settings.set_string("/persistent/xr/anchorMode", "custom anchor")
            settings.set_string("/xrstage/customAnchor", self._xr_anchor_headset_path)

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


class ExternalXrAnchor:
    """XR anchor of a session whose renderer and Stage belong to the application (OVXR).

    Presents the interface :class:`~isaaclab_teleop.IsaacTeleopDevice` consumes from
    :class:`XrAnchorManager`, without reading or configuring Kit's XRCore. A static anchor
    (``XrCfg.anchor_prim_path`` unset) keeps its configured pose. A dynamic anchor follows the
    reference prim and implements ``OvLibrariesXrSession.ExternalAnchorState``: the session host
    commits each pose together with its scene publication, and :meth:`get_world_matrix` raises
    until the current reset epoch has a committed pose.
    """

    def __init__(self, xr_cfg: XrCfg):
        """Create or refresh the stage-scoped anchor prim at its configured world pose.

        Reusing the Stage across sessions does not retain a previous session's anchor configuration.

        Args:
            xr_cfg: XR configuration specifying anchor position, rotation,
                and optional dynamic anchoring prim path.

        Raises:
            ValueError: An existing anchor prim is not transformable. Other anchor authoring
                errors also propagate, so rendering cannot continue with a pose that differs
                from tracking.
        """
        self._xr_cfg = xr_cfg
        reference_path = xr_cfg.anchor_prim_path
        self._anchor_headset_path = (
            f"{reference_path.removesuffix('/')}/XRAnchor" if reference_path is not None else "/World/XRAnchor"
        )
        # Anchors survive per-run device teardown. Reuse their paths while
        # applying this run's configuration, rather than retaining a stale pose.
        _create_or_update_xr_anchor(self._anchor_headset_path, xr_cfg.anchor_pos, xr_cfg.anchor_rot)

        self._lock = threading.RLock()
        self._policy = _XrAnchorPosePolicy(xr_cfg) if reference_path is not None else None
        self._epoch = 0
        self._error = None
        # world_T_anchor; a dynamic anchor has none until the host commits one for the epoch.
        self._world_matrix = None
        if self._policy is None:
            self._world_matrix = np.eye(4)
            self._world_matrix[:3, :3] = Rotation.from_quat(xr_cfg.anchor_rot).as_matrix()
            self._world_matrix[:3, 3] = xr_cfg.anchor_pos

    @property
    def anchor_headset_path(self) -> str:
        """The USD path of the XR anchor prim."""
        return self._anchor_headset_path

    def get_world_matrix(self) -> np.ndarray:
        """Build the combined 4x4 transform from OpenXR local space to Isaac Lab world.

        Returns:
            A (4, 4) float32 numpy array; see :meth:`XrAnchorManager.get_world_matrix`.

        Raises:
            RuntimeError: A dynamic anchor has no committed pose for the current reset epoch,
                or a previous anchor publication failed.
        """
        with self._lock:
            self._raise_if_failed()
            if self._world_matrix is None:
                raise RuntimeError("Dynamic XR anchor requires a committed post-reset scene pose")
            matrix = self._world_matrix.copy()
        matrix[:3, :3] = matrix[:3, :3] @ XrAnchorManager._OXR_TO_USD_ROTATION
        return matrix.astype(np.float32)

    def reset(self) -> None:
        """Start a new reset epoch; a dynamic anchor then needs a newly committed pose."""
        if self._policy is not None:
            with self._lock:
                self._epoch += 1
                self._policy.reset()
                self._world_matrix = None

    def toggle_anchor_rotation(self) -> None:
        """Toggle anchor rotation following of a dynamic anchor."""
        if self._policy is not None:
            with self._lock:
                self._policy.toggle_anchor_rotation()

    def cleanup(self) -> None:
        """Nothing to release; the anchor holds no subscriptions."""

    @property
    def external_reference_path(self) -> str | None:
        """The exact physics reference path of a dynamic anchor, or ``None`` for a static anchor."""
        return self._xr_cfg.anchor_prim_path if self._policy is not None else None

    @property
    def external_epoch(self) -> tuple[int, bool]:
        """Return reset identity and whether the current epoch needs its first publication."""
        with self._lock:
            self._raise_if_failed()
            return self._epoch, self._world_matrix is None

    @property
    def external_anchor_paths(self) -> tuple[str, ...]:
        """Native Xforms written together by the OVXR presentation host."""
        return (self._anchor_headset_path,)

    def commit_external_pose(self, reference, rendering_dt: float, epoch: int, commit) -> bool:
        """Advance the shared policy once and expose its input pose only after native seal.

        The host serializes this operation with scene publication. Old committed
        references cannot reseed a new reset epoch. A failed write invalidates the
        cached input pose and propagates; it never falls back to configured static data.
        """
        with self._lock:
            self._raise_if_failed()
            if epoch != self._epoch:
                return False
            try:
                matrix = np.asarray(self._policy.update(reference, rendering_dt)).T.copy()
                commit(matrix[np.newaxis])
            except BaseException as exc:
                self._world_matrix = None
                self._error = exc
                raise
            self._world_matrix = matrix
            return True

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Dynamic XR anchor publication failed; recreate the session") from self._error
