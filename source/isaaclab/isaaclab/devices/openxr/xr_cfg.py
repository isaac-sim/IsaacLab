# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class XrCfg:
    """Configuration for viewing and interacting with the environment through an XR device."""

    anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Specifies the position (in m) of the simulation when viewed in an XR device.

    Specifically: this position will appear at the origin of the XR device's local coordinate frame.
    """

    anchor_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Specifies the rotation (as a quaternion) of the simulation when viewed in an XR device.

    Specifically: this rotation will determine how the simulation is rotated with respect to the
    origin of the XR device's local coordinate frame.

    This quantity is only effective if :attr:`xr_anchor_pos` is set.
    """

    near_plane: float = 0.15
    """Specifies the near plane distance for the XR device.

    This value determines the closest distance at which objects will be rendered in the XR device.
    """


from typing import Any


def remove_camera_configs(env_cfg: Any) -> Any:
    """Removes cameras from environments when using XR devices.

    XR does not support additional cameras in the environment as they can cause
    rendering conflicts and performance issues. This function scans the environment
    configuration for camera objects and removes them, along with any associated
    observation terms that reference these cameras.

    Args:
        env_cfg: The environment configuration to modify.

    Returns:
        The modified environment configuration with cameras removed.
    """

    import omni.log

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    for attr_name in dir(env_cfg.scene):
        attr = getattr(env_cfg.scene, attr_name)
        if isinstance(attr, CameraCfg):
            delattr(env_cfg.scene, attr_name)
            omni.log.info(f"Removed camera config: {attr_name}")

            # Remove any ObsTerms for the camera
            if hasattr(env_cfg.observations, "policy"):
                for obs_name in dir(env_cfg.observations.policy):
                    obsterm = getattr(env_cfg.observations.policy, obs_name)
                    if hasattr(obsterm, "params") and obsterm.params:
                        for param_value in obsterm.params.values():
                            if isinstance(param_value, SceneEntityCfg) and param_value.name == attr_name:
                                delattr(env_cfg.observations.policy, attr_name)
                                omni.log.info(f"Removed camera observation term: {attr_name}")
                                break
    return env_cfg
