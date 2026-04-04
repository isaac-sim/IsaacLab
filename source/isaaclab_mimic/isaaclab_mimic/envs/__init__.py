# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_stack_ik_rel_mimic_env_cfg:FrankaCubeStackIKRelMimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    entry_point=f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.franka_stack_ik_rel_blueprint_mimic_env_cfg:FrankaCubeStackIKRelBlueprintMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.franka_stack_ik_abs_mimic_env:FrankaCubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_stack_ik_abs_mimic_env_cfg:FrankaCubeStackIKAbsMimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0",
    entry_point=f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.franka_stack_ik_rel_visuomotor_mimic_env_cfg:FrankaCubeStackIKRelVisuomotorMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0",
    entry_point=f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_stack_ik_rel_visuomotor_cosmos_mimic_env_cfg:FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg",
    },
    disable_env_checker=True,
)


##
# SkillGen
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    entry_point=f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_stack_ik_rel_skillgen_env_cfg:FrankaCubeStackIKRelSkillgenEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_bin_stack_ik_rel_mimic_env_cfg:FrankaBinStackIKRelMimicEnvCfg",
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

##
# GR1T2 Pick Place with Pink IK - Absolute Pose Control
##

gym.register(
    id="Isaac-PickPlace-GR1T2-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_mimic_env_cfg:PickPlaceGR1T2MimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pickplace_gr1t2_waist_enabled_mimic_env_cfg:PickPlaceGR1T2WaistEnabledMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-NutPour-GR1T2-Pink-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.nutpour_gr1t2_mimic_env_cfg:NutPourGR1T2MimicEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.exhaustpipe_gr1t2_mimic_env_cfg:ExhaustPipeGR1T2MimicEnvCfg"},
    disable_env_checker=True,
)

##
# Locomanipulation G1 with Pink IK - Absolute Pose Control
##

gym.register(
    id="Isaac-Locomanipulation-G1-Abs-Mimic-v0",
    entry_point=f"{__name__}.locomanipulation_g1_mimic_env:LocomanipulationG1MimicEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.locomanipulation_g1_mimic_env_cfg:LocomanipulationG1MimicEnvCfg"},
    disable_env_checker=True,
)
