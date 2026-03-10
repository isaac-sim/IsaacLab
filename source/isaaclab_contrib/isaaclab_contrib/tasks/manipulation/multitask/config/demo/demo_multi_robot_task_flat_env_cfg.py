# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat (hand-written) multi-robot multi-task env config.

Demonstrates how to write a heterogeneous multi-robot environment *without*
the ``MultiTaskRegistryConfig`` composition machinery.  Every scene asset,
action term, and prim-path is specified explicitly so the user has full control.

Per-group assets use ``{ENV_REGEX_NS}`` in their ``prim_path`` like shared
assets, but set :attr:`~isaaclab.assets.AssetBaseCfg.assigned_env_ids` so
that the cloner only replicates them into the environments assigned to their
task group.  The ``assigned_env_ids`` are populated in ``__post_init__``
after the environment partition is computed.

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

TASK_OPENARM_LIFT = 0
TASK_FRANKA_STACK = 1
TASK_UR10_REACH = 2

NUM_GROUPS = 3
NUM_ENVS = 24


def _partition_env_ids(num_envs: int, num_groups: int) -> list[list[int]]:
    """Split *num_envs* indices as evenly as possible across *num_groups*."""
    base, remainder = divmod(num_envs, num_groups)
    groups: list[list[int]] = []
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


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

    All per-group assets use ``{ENV_REGEX_NS}`` in their ``prim_path``.
    :attr:`~isaaclab.assets.AssetBaseCfg.assigned_env_ids` is set in
    ``FlatMultiRobotLiftStackEnvCfg.__post_init__`` so that the cloner
    only replicates each asset into the environments assigned to its group.
    """

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
    # assigned_env_ids set in __post_init__
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/OpenArm_Robot")
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    openarm_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_LIFT_CUBE_SPAWN,
    )

    # -- Group 1: Franka Stack (differential-IK) ------------------------------
    # assigned_env_ids set in __post_init__
    franka_stack_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka_Robot")
    franka_stack_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    franka_cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_1_SPAWN,
    )
    franka_cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.05, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_2_SPAWN,
    )
    franka_cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.1, 0.0203), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_STACK_CUBE_3_SPAWN,
    )

    # -- Group 2: UR10 Reach (IK controller, no gripper, no objects) ----------
    # assigned_env_ids set in __post_init__
    ur10_reach_robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/UR10_Robot")
    ur10_reach_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
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


    ``__post_init__`` partitions ``num_envs`` across groups and sets
    ``assigned_env_ids`` on every per-group scene asset so the cloner
    only replicates each asset into the environments it belongs to.

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

        groups = _partition_env_ids(self.scene.num_envs, NUM_GROUPS)

        # Group 0: OpenArm lift
        for asset in (self.scene.openarm_robot, self.scene.openarm_table, self.scene.openarm_cube):
            asset.assigned_env_ids = groups[TASK_OPENARM_LIFT]

        # Group 1: Franka stack
        for asset in (
            self.scene.franka_stack_robot,
            self.scene.franka_stack_table,
            self.scene.franka_cube_1,
            self.scene.franka_cube_2,
            self.scene.franka_cube_3,
        ):
            asset.assigned_env_ids = groups[TASK_FRANKA_STACK]

        # Group 2: UR10 reach
        for asset in (self.scene.ur10_reach_robot, self.scene.ur10_reach_table):
            asset.assigned_env_ids = groups[TASK_UR10_REACH]
