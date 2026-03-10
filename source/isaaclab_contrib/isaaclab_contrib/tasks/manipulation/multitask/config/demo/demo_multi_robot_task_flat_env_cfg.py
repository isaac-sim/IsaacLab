# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat (hand-written) multi-robot multi-task env config.

Demonstrates how to write a heterogeneous multi-robot environment *without*
the ``MultiTaskRegistryConfig`` composition machinery.  Every scene asset,
action term, and prim-path is specified explicitly so the user has full control.

Per-group assets declare ``task_group`` to indicate which task group they
belong to.  The scene's ``task_groups`` dict defines the groups and their
relative weights; env-id partitioning and cloning masks are resolved
automatically by the :class:`InteractiveScene`.

Layout (3 groups, evenly split):
    Group 0:  OpenArm  -- Lift Cube          (differential-IK actions)
    Group 1:  Franka   -- Stack Cubes        (differential-IK actions)
    Group 2:  UR10     -- Reach (no objects)  (differential-IK actions)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

from .demo_multitask_flat_env_cfg import MultitaskPhysicsCfg

TASK_OPENARM_LIFT = "openarm_lift"
TASK_FRANKA_STACK = "franka_stack"
TASK_UR10_REACH = "ur10_reach"

NUM_ENVS = 25


_LIFT_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_LIFT_CUBE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    scale=(0.8, 0.8, 0.8),
    rigid_props=_LIFT_CUBE_RIGID_PROPS,
)

_STACK_CUBE_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

_STACK_CUBE_1_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_STACK_CUBE_2_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_STACK_CUBE_3_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
    scale=(1.0, 1.0, 1.0),
    rigid_props=_STACK_CUBE_RIGID_PROPS,
)

_TABLE_SPAWN = UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")


@configclass
class FlatMultiRobotSceneCfg(InteractiveSceneCfg):
    """Scene with three robot groups, each in its own env-id subset.

    All per-group assets declare ``task_group`` to specify which group
    they belong to.  The ``task_groups`` dict defines the partition.
    """

    # Values are relative weights: 1:1:1 means equal split across the three groups.
    task_groups = {TASK_OPENARM_LIFT: 1, TASK_FRANKA_STACK: 1, TASK_UR10_REACH: 1}

    # -- shared (all envs) ---------------------------------------------------
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # -- Group 0: OpenArm Lift -----------------------------------------------
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot", task_group=TASK_OPENARM_LIFT
    )
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
        task_group=TASK_OPENARM_LIFT,
    )
    openarm_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_LIFT_CUBE_SPAWN,
        task_group=TASK_OPENARM_LIFT,
    )

    # -- Group 1: Franka Stack (differential-IK) ------------------------------
    franka_stack_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot", task_group=TASK_FRANKA_STACK
    )
    franka_stack_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
        task_group=TASK_FRANKA_STACK,
    )
    franka_cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_1_SPAWN,
        task_group=TASK_FRANKA_STACK,
    )
    franka_cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.05, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_2_SPAWN,
        task_group=TASK_FRANKA_STACK,
    )
    franka_cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.1, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_3_SPAWN,
        task_group=TASK_FRANKA_STACK,
    )

    # -- Group 2: UR10 Reach (IK controller, no gripper, no objects) ----------
    ur10_reach_robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/UR10_Robot", task_group=TASK_UR10_REACH)
    ur10_reach_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
        task_group=TASK_UR10_REACH,
    )


@configclass
class FlatMultiRobotActionsCfg:
    """Per-group action terms.

    All three groups use differential inverse kinematics for the arm.
    Groups 0 & 1 also have binary gripper actions.
    """

    # Group 0: OpenArm (differential IK)
    openarm_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_joint.*"],
        body_name="openarm_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )
    openarm_gripper = BinaryJointPositionActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )

    # Group 1: Franka Stack (differential IK)
    franka_stack_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="franka_stack_robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    franka_stack_gripper = BinaryJointPositionActionCfg(
        asset_name="franka_stack_robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )

    # Group 2: UR10 Reach (differential IK, no gripper)
    ur10_reach_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="ur10_reach_robot",
        joint_names=[".*"],
        body_name="ee_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )


@configclass
class FlatCommandsCfg:
    """Command terms – reach target for the UR10 end-effector.

    The command is generated for every env in Group 2 (UR10_REACH) task.
    """

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="ur10_reach_robot",
        body_name="ee_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class FlatObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        sim_time_s = ObsTerm(func=mdp.current_time_s)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FlatRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class FlatTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FlatEventsCfg:
    reset_scene_to_default = EventTerm(func=mdp.reset_multitask_scene_to_default, mode="reset")


@configclass
class FlatMultiRobotMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Hand-written multi-robot env with three heterogeneous groups.

    Task groups and per-asset ``task_group`` declarations are defined in
    the scene config.  Env-id partitioning is handled automatically by
    :class:`InteractiveScene`.

    Group 0: OpenArm  -- lift cube          (differential-IK actions)
    Group 1: Franka   -- stack 3 cubes      (differential-IK actions)
    Group 2: UR10     -- reach (no objects)  (differential-IK actions)
    """

    scene: FlatMultiRobotSceneCfg = FlatMultiRobotSceneCfg(num_envs=NUM_ENVS, env_spacing=2.0, replicate_physics=False)
    actions: FlatMultiRobotActionsCfg = FlatMultiRobotActionsCfg()
    commands: FlatCommandsCfg = FlatCommandsCfg()
    observations: FlatObservationsCfg = FlatObservationsCfg()
    rewards: FlatRewardsCfg = FlatRewardsCfg()
    terminations: FlatTerminationsCfg = FlatTerminationsCfg()
    events: FlatEventsCfg = FlatEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 3
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()
