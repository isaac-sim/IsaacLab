# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import enum
from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass


class XrAnchorRotationMode(enum.Enum):
    """Enumeration for XR anchor rotation modes."""

    FIXED = "fixed"
    """Fixed rotation mode: sets rotation once and doesn't change it."""

    FOLLOW_PRIM = "follow_prim"
    """Follow prim rotation mode: rotation follows prim's rotation."""

    FOLLOW_PRIM_SMOOTHED = "follow_prim_smoothed"
    """Follow prim rotation mode with smooth interpolation: rotation smoothly follows prim's rotation using slerp."""

    CUSTOM = "custom_rotation"
    """Custom rotation mode: user provided function to calculate the rotation."""


@configclass
class XrCfg:
    """Configuration for viewing and interacting with the environment through an XR device."""

    anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Specifies the position (in m) of the simulation when viewed in an XR device.

    Specifically: this position will appear at the origin of the XR device's local coordinate frame.
    """

    anchor_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Specifies the rotation (as a quaternion xyzw) of the simulation when viewed in an XR device.

    Specifically: this rotation will determine how the simulation is rotated with respect to the
    origin of the XR device's local coordinate frame.

    This quantity is only effective if :attr:`xr_anchor_pos` is set.
    """

    render_anchor_pos: tuple[float, float, float] | None = None
    """Optional position (in m) used only to frame the XR view.

    When ``None``, rendering and tracked-device poses share :attr:`anchor_pos`,
    which preserves the historical behavior. Set this when the operator view
    needs a different origin from the world frame used for teleoperation.
    """

    render_anchor_rot: tuple[float, float, float, float] | None = None
    """Optional render-only rotation as a quaternion in ``xyzw`` order.

    When ``None``, the render anchor uses :attr:`anchor_rot`. This field is
    effective only when either render-anchor override is configured.
    """

    hidden_visual_prims: list[str] = []
    """Prim-path regex patterns hidden from the kitless OVXR view.

    Tasks can use this to hide robot geometry that encloses the eye cameras.
    The OVXR session authors visibility on the live stage before
    renderer materialization and fails if a configured pattern matches no
    imageable prim. Visibility persists for the application Stage's lifetime
    and affects all cameras, including PiP; it is not restored on XR close.
    """

    excluded_visual_meshes: list[str] = []
    """Mesh-path regexes excluded from kitless environment cameras by scene partition.

    Unlike ``hidden_visual_prims``, this leaves USD visibility and the source
    scene unchanged. Only matching visual meshes in the single-environment
    render export enter the ``xr_excluded`` partition. Camera selectors remain
    independent; collisions and link/camera transforms must not be matched.
    """

    anchor_prim_path: str | None = None
    """Specifies the prim path to attach the XR anchor to for dynamic positioning.

    When set, the XR anchor will be attached to the specified prim (e.g., robot root prim),
    allowing the XR camera to move with the prim. This is particularly useful for locomotion
    robot teleoperation where the robot moves and the XR camera should follow it.

    Kit sessions use XRCore to follow the reference. Application-owned OVXR presentation
    uses committed physics references for the same rotation and height policies. Input cannot
    consume an unseeded or failed publication epoch; no static pose is substituted.

    If None, the anchor will use the static :attr:`anchor_pos` and :attr:`anchor_rot` values.
    """

    anchor_rotation_mode: XrAnchorRotationMode = XrAnchorRotationMode.FIXED
    """Specifies how the XR anchor rotation should behave when attached to a prim.

    The available modes are:
    - :attr:`XrAnchorRotationMode.FIXED`: Sets rotation once to anchor_rot value
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM`: Rotation follows prim's rotation
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED`: Rotation smoothly follows prim's rotation using slerp
    - :attr:`XrAnchorRotationMode.CUSTOM`: user provided function to calculate the rotation
    """

    anchor_rotation_smoothing_time: float = 1.0
    """Wall-clock time constant (seconds) for rotation smoothing in FOLLOW_PRIM_SMOOTHED mode.

    This time constant is applied using wall-clock delta time between frames (not physics dt).
    Smaller values (e.g., 0.1) result in faster/snappier response but less smoothing.
    Larger values (e.g., 0.75–2.0) result in slower/smoother response but more lag.
    Typical useful range: 0.3 – 1.5 seconds depending on runtime frame-rate and comfort.
    """

    anchor_rotation_custom_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda headpose, primpose: np.array(
        [0, 0, 0, 1], dtype=np.float64
    )
    """Specifies the function to calculate the rotation of the XR anchor when anchor_rotation_mode is CUSTOM.

    Args:
        headpose: Previous head pose as numpy array [x, y, z, w, x, y, z] (position + quaternion)
        pose: Anchor prim pose as numpy array [x, y, z, w, x, y, z] (position + quaternion)

    Returns:
        np.ndarray: Quaternion as numpy array [w, x, y, z]
    """

    near_plane: float = 0.15
    """Specifies the near plane distance for the XR device.

    This value determines the closest distance at which objects will be rendered in the XR device.
    """

    fixed_anchor_height: bool = True
    """Specifies if the anchor height should be fixed.

    If True, the anchor height will be fixed to the initial height of the anchor prim.
    """


from typing import Any


def remove_camera_configs(env_cfg: Any, *, keep: tuple[str, ...] = ()) -> Any:
    """Removes cameras from environments when using XR devices.

    Having additional cameras cause operation performance issues. This function scans the environment
    configuration for camera objects and removes them, along with any associated
    observation terms that reference these cameras.

    Args:
        env_cfg: The environment configuration to modify.
        keep: Camera attribute names to retain for the OVXR renderer.
            Observation terms are removed for retained cameras as well, so the
            application loop never requests a second camera render.

    Returns:
        The modified environment configuration with cameras removed.
    """

    import logging

    # import logger
    logger = logging.getLogger(__name__)

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    for attr_name in dir(env_cfg.scene):
        attr = getattr(env_cfg.scene, attr_name)
        if isinstance(attr, CameraCfg):
            if attr_name not in keep:
                delattr(env_cfg.scene, attr_name)
                logger.info(f"Removed camera config: {attr_name}")

            # Remove any ObsTerms for the camera
            if hasattr(env_cfg.observations, "policy"):
                for obs_name in dir(env_cfg.observations.policy):
                    obsterm = getattr(env_cfg.observations.policy, obs_name)
                    if hasattr(obsterm, "params") and obsterm.params:
                        for param_value in obsterm.params.values():
                            if isinstance(param_value, SceneEntityCfg) and param_value.name == attr_name:
                                delattr(env_cfg.observations.policy, obs_name)
                                logger.info(f"Removed camera observation term: {obs_name}")
                                break
    return env_cfg
