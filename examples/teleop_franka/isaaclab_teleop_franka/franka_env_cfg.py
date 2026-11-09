# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attach keyboard or Quest controller input to the original Franka Reach actions.

Select ``presets=diffik,keyboard`` for keyboard control or
``presets=diffik_abs,xr`` for calibrated absolute pose control from Quest.
Without these selections the task retains its original joint-position controller.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab_teleop import (
    CLOUDXR_JS_ENV,
    CaptureTeleopSourceCfg,
    IsaacTeleopCfg,
    OvLibrariesXrCfg,
    TeleopInputCfg,
    TeleopObservationCfg,
    XrCfg,
    se3_keyboard_teleop_cfg,
)

from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import FrankaArmActionCfg, FrankaReachEnvCfg
from isaaclab_tasks.utils import preset

if TYPE_CHECKING:
    from isaaccapture.retargeting_engine.interface import OutputCombiner


def _controller_pose_pipeline() -> OutputCombiner:
    """Load the Capture SDK only when the XR input source starts."""
    from .xr import controller_pose_pipeline

    return controller_pose_pipeline()


@configclass
class FrankaTeleopArmActionCfg(FrankaArmActionCfg):
    """Preserve the task's controllers and add independently selectable inputs."""

    diffik = replace(
        FrankaArmActionCfg().diffik,
        input=preset(
            default=None,
            keyboard=TeleopInputCfg(
                source=CaptureTeleopSourceCfg(
                    device=se3_keyboard_teleop_cfg(
                        # The existing action scales these to 5 mm and 0.01 rad per step.
                        pos_sensitivity=0.1,
                        rot_sensitivity=0.02,
                        gripper_term=False,
                    ),
                ),
                env_ids=[0],
            ),
        ),
    )
    diffik_abs = replace(
        FrankaArmActionCfg().diffik_abs,
        input=preset(
            default=None,
            xr=TeleopInputCfg(
                class_type="isaaclab_teleop_franka.xr:CalibratedAbsolutePoseInput",
                source=CaptureTeleopSourceCfg(
                    device=IsaacTeleopCfg(
                        pipeline_builder=_controller_pose_pipeline,
                        xr_cfg=XrCfg(
                            anchor_pos=(-0.65, 0.0, -1.05),
                            anchor_rot=(0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)),
                        ),
                        teleoperation_active_default=False,
                        app_name="IsaacLabFrankaReachXR",
                    ),
                    start_active=False,
                ),
                env_ids=[0],
            ),
        ),
    )


@configclass
class FrankaTeleopEnvCfg(FrankaReachEnvCfg):
    """Keep Franka Reach physics, observations, rewards, resets, and agent interfaces."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action = FrankaTeleopArmActionCfg()
        self.observations.operator = preset(
            default=None,
            xr=TeleopObservationCfg(
                presentation=OvLibrariesXrCfg(
                    application_name="IsaacLabFrankaReachXR",
                    session_mode="vr",
                    cloudxr_env_file=CLOUDXR_JS_ENV,
                    desktop_mirror_enabled=False,
                ),
                # Initial RTX shader compilation can take several minutes.
                startup_timeout_s=600.0,
            ),
        )

    def validate_config(self) -> None:
        super().validate_config()
        if self.observations.operator is not None and (
            not isinstance(self.actions.arm_action.input, TeleopInputCfg)
            or self.actions.arm_action.input.class_type != "isaaclab_teleop_franka.xr:CalibratedAbsolutePoseInput"
        ):
            raise ValueError("The Franka XR example requires presets=diffik_abs,xr.")
