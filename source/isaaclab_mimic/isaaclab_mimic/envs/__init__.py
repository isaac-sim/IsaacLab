# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .franka_bin_stack_ik_rel_mimic_env_cfg import FrankaBinStackIKRelMimicEnvCfg
from .franka_stack_ik_abs_mimic_env import FrankaCubeStackIKAbsMimicEnv
from .franka_stack_ik_abs_mimic_env_cfg import FrankaCubeStackIKAbsMimicEnvCfg
from .franka_stack_ik_rel_blueprint_mimic_env_cfg import FrankaCubeStackIKRelBlueprintMimicEnvCfg
from .franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv
from .franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from .franka_stack_ik_rel_skillgen_env_cfg import FrankaCubeStackIKRelSkillgenEnvCfg
from .franka_stack_ik_rel_visuomotor_cosmos_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg
from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_mimic_env_cfg.FrankaCubeStackIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_blueprint_mimic_env_cfg.FrankaCubeStackIKRelBlueprintMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_abs_mimic_env_cfg.FrankaCubeStackIKAbsMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            franka_stack_ik_rel_visuomotor_cosmos_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg
        ),
    },
    disable_env_checker=True,
)


##
# SkillGen
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_skillgen_env_cfg.FrankaCubeStackIKRelSkillgenEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_bin_stack_ik_rel_mimic_env_cfg.FrankaBinStackIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)

##
# Galbot Stack Cube with RmpFlow - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_rel_mimic_env:RmpFlowGalbotCubeStackRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_rel_mimic_env_cfg:RmpFlowGalbotLeftArmGripperCubeStackRelMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_rel_mimic_env:RmpFlowGalbotCubeStackRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_rel_mimic_env_cfg:RmpFlowGalbotRightArmSuctionCubeStackRelMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

##
# Galbot Stack Cube with RmpFlow - Absolute Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-Abs-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_abs_mimic_env:RmpFlowGalbotCubeStackAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_abs_mimic_env_cfg:RmpFlowGalbotLeftArmGripperCubeStackAbsMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-Abs-Mimic-v0",
    entry_point=f"{__name__}.galbot_stack_rmp_abs_mimic_env:RmpFlowGalbotCubeStackAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.galbot_stack_rmp_abs_mimic_env_cfg:RmpFlowGalbotRightArmSuctionCubeStackAbsMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

##
# Agibot Left Arm: Place Upright Mug with RmpFlow - Relative Pose Control
##
gym.register(
    id="Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.agibot_place_upright_mug_mimic_env_cfg:RmpFlowAgibotPlaceUprightMugMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
##
# Agibot Right Arm: Place Toy2Box: RmpFlow - Relative Pose Control
##
gym.register(
    id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0",
    entry_point=f"{__name__}.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.agibot_place_toy2box_mimic_env_cfg:RmpFlowAgibotPlaceToy2BoxMimicEnvCfg",
    },
    disable_env_checker=True,
)
