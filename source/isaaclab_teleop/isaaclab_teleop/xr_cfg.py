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
    """Follow the prim's yaw change relative to its initial orientation, composed with :attr:`XrCfg.anchor_rot`."""

    FOLLOW_PRIM_SMOOTHED = "follow_prim_smoothed"
    """Smoothly follow the prim's relative yaw change, composed with :attr:`XrCfg.anchor_rot`, using slerp."""

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

    This rotation also applies when no :attr:`anchor_prim_path` is set.
    """

    anchor_prim_path: str | None = None
    """Specifies the prim path to attach the XR anchor to for dynamic positioning.

    When set, the XR anchor will be attached to the specified prim (e.g., robot root prim),
    allowing the XR camera to move with the prim. This is particularly useful for locomotion
    robot teleoperation where the robot moves and the XR camera should follow it.
    :func:`~isaaclab_teleop.create_isaac_teleop_device` expands ``{ENV_REGEX_NS}`` when
    given an ``env_prim_path``; direct device construction requires a concrete path.

    If None, the anchor will use the static :attr:`anchor_pos` and :attr:`anchor_rot` values.
    """

    anchor_rotation_mode: XrAnchorRotationMode = XrAnchorRotationMode.FIXED
    """Specifies how the XR anchor rotation should behave when attached to a prim.

    The available modes are:

    - :attr:`XrAnchorRotationMode.FIXED`: Sets rotation once to anchor_rot value
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM`: Follows the prim's yaw change relative
      to its initial orientation, composed with :attr:`anchor_rot`; roll and pitch are ignored
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED`: Smoothly follows the same
      relative yaw target using slerp
    - :attr:`XrAnchorRotationMode.CUSTOM`: user provided function to calculate the rotation
    """

    anchor_rotation_smoothing_time: float = 1.0
    """Time constant [s] for rotation smoothing in FOLLOW_PRIM_SMOOTHED mode.

    Smoothing uses :meth:`isaaclab.sim.SimulationContext.get_rendering_dt`, which returns
    the configured rendering timestep or a visualizer override. The interpolation factor is
    clamped to [0.05, 1.0] per update.
    Smaller values (e.g., 0.1) result in faster/snappier response but less smoothing.
    Larger values (e.g., 0.75–2.0) result in slower/smoother response but more lag.
    Typical useful range: 0.3 – 1.5 seconds depending on rendering timestep and comfort.
    """

    anchor_rotation_custom_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda headpose, primpose: np.array(
        [0, 0, 0, 1], dtype=np.float64
    )
    """Specifies the function to calculate the rotation of the XR anchor when anchor_rotation_mode is CUSTOM.

    Args:
        headpose: Previous head pose as a NumPy array [x, y, z, qx, qy, qz, qw]
            (position [m] and xyzw quaternion).
        primpose: Anchor prim pose as a NumPy array [x, y, z, qx, qy, qz, qw]
            (position [m] and xyzw quaternion).

    Returns:
        Quaternion as a NumPy array [qx, qy, qz, qw] (xyzw).
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


def remove_camera_configs(env_cfg: Any) -> Any:
    """Removes cameras from environments when using XR devices.

    Having additional cameras cause operation performance issues. This function scans the environment
    configuration for camera objects and removes them, along with any associated
    observation terms that reference these cameras.

    Args:
        env_cfg: The environment configuration to modify.

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
