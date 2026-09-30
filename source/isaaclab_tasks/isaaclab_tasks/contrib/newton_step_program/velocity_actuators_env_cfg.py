# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat ANYmal-D velocity tracking with actuator mixes that exercise captured and eager Newton actuators."""

from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg

from isaaclab_assets.robots.anymal import ANYDRIVE_3_LSTM_ACTUATOR_CFG, ANYDRIVE_3_SIMPLE_ACTUATOR_CFG


@configclass
class AnymalDDCMotorEnvCfg(AnymalDFlatEnvCfg):
    """Flat ANYmal-D with DC-motor drives, whose Newton actuators the step program captures."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = {"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG}


@configclass
class AnymalDMixedActuatorsEnvCfg(AnymalDFlatEnvCfg):
    """Flat ANYmal-D with DC motors on the hind legs and LSTM networks on the front legs.

    The TorchScript networks cannot be captured, so the step program runs them eagerly between captured segments.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = {
            "hind_legs": replace(
                ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, joint_names_expr=["[LR]H_HAA", "[LR]H_HFE", "[LR]H_KFE"]
            ),
            "front_legs": replace(
                ANYDRIVE_3_LSTM_ACTUATOR_CFG, joint_names_expr=["[LR]F_HAA", "[LR]F_HFE", "[LR]F_KFE"]
            ),
        }
