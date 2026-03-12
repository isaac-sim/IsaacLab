# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot reach env: three robot types, all reach.

Each robot type occupies its own env-id group.  The
:class:`ActionManager` automatically shares action columns
across disjoint groups, so three DiffIK terms (each dim=6)
produce ``total_action_dim = 6`` instead of 18.

Observations are task-space-centric: EE pose, command target,
and position error are the same dimension for all robots
(no padding).  Joint-space terms are auto-padded to
``layout.max_arm_dof``.

Layout (3 groups, evenly split):
    Group 0:  OpenArm -- Reach (7 arm DoF)
    Group 1:  Franka  -- Reach (7 arm DoF)
    Group 2:  UR10    -- Reach (6 arm DoF)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import (
    DifferentialIKControllerCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp

from isaaclab_tasks.manager_based.manipulation.reach.mdp import rewards as reach_rewards

from isaaclab_assets.robots.franka import (
    FRANKA_PANDA_HIGH_PD_CFG,
)
from isaaclab_assets.robots.openarm import (
    OPENARM_UNI_HIGH_PD_CFG,
)
from isaaclab_assets.robots.universal_robots import UR10_CFG

from .demo_multitask_flat_env_cfg import MultitaskPhysicsCfg

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

TASK_OPENARM = "openarm_reach"
TASK_FRANKA = "franka_reach"
TASK_UR10 = "ur10_reach"

_TABLE_SPAWN = UsdFileCfg(
    usd_path=(f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
)


# ===========================================================
# Scene
# ===========================================================


@configclass
class MultiRobotReachSceneCfg(InteractiveSceneCfg):
    """Three robot types, all doing reach (no objects)."""

    task_groups = {
        TASK_OPENARM: 1,
        TASK_FRANKA: 1,
        TASK_UR10: 1,
    }

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -1.05),
        ),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
        task_group=TASK_OPENARM,
    )
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=_TABLE_SPAWN,
        task_group=TASK_OPENARM,
    )
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
        task_group=TASK_FRANKA,
    )
    franka_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=_TABLE_SPAWN,
        task_group=TASK_FRANKA,
    )
    ur10_robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UR10_Robot",
        task_group=TASK_UR10,
    )
    ur10_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10_Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
        spawn=_TABLE_SPAWN,
        task_group=TASK_UR10,
    )


# ===========================================================
# Actions  (3 DiffIK terms, columns shared → action_dim=6)
# ===========================================================


_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@configclass
class MultiRobotReachActionsCfg:
    """Three DiffIK terms (one per robot group).

    Because the ActionManager shares action columns across
    disjoint groups, total_action_dim = max(6, 6, 6) = 6.
    """

    openarm_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_joint.*"],
        body_name="openarm_hand",
        controller=_IK_CTRL,
        scale=0.5,
    )
    franka_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="franka_robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=_IK_CTRL,
        scale=0.5,
    )
    ur10_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="ur10_robot",
        joint_names=[".*"],
        body_name="ee_link",
        controller=_IK_CTRL,
        scale=0.5,
    )


# ===========================================================
# Commands
# ===========================================================


@configclass
class MultiRobotReachCommandsCfg:
    """Per-group reach command targets."""

    openarm_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="openarm_robot",
        body_name="openarm_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.35),
            pos_y=(-0.2, 0.2),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 9, math.pi / 9),
        ),
    )
    franka_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="franka_robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
    )
    ur10_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="ur10_robot",
        body_name="ee_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-3.14, 3.14),
        ),
    )


# ===========================================================
# Observations
# ===========================================================


@configclass
class MultiRobotReachObsCfg:
    """Task-space + proprioceptive observations.

    Terms with ``per_robot=True`` reuse standard observation functions;
    the manager auto-injects ``asset_cfg`` and ``command_name`` from
    each :class:`RobotSpec` and scatters results (with zero-padding)
    into a single ``(num_envs, max_feat)`` tensor.

    Task-space terms (EE pose, command, error) have the same
    dimension regardless of robot DoF.  Joint-space terms are
    auto-padded to the maximum across robot groups.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        robot_type = ObsTerm(func=mdp.multi_robot_type_onehot)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, per_robot=True)
        joint_vel = ObsTerm(func=mdp.joint_vel, per_robot=True)
        ee_pose = ObsTerm(func=mdp.ee_pose_b, per_robot=True)
        ee_command = ObsTerm(func=mdp.generated_commands, per_robot=True)
        ee_pos_error = ObsTerm(func=mdp.ee_pos_error, per_robot=True)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===========================================================
# Rewards / Terminations / Events
# ===========================================================


@configclass
class MultiRobotReachRewardsCfg:
    """Reach rewards auto-dispatched across all robot groups.

    Terms with ``per_robot=True`` reuse standard reward functions;
    the manager auto-injects ``asset_cfg`` and ``command_name``
    from each :class:`RobotSpec` and scatters results into a
    single ``(num_envs,)`` tensor.
    """

    ee_pos_tracking = RewTerm(
        func=reach_rewards.position_command_error,
        weight=-0.2,
        per_robot=True,
    )
    ee_pos_tracking_fine = RewTerm(
        func=reach_rewards.position_command_error_tanh,
        weight=0.1,
        per_robot=True,
        params={"std": 0.1},
    )
    ee_ori_tracking = RewTerm(
        func=reach_rewards.orientation_command_error,
        weight=-0.1,
        per_robot=True,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        per_robot=True,
    )


@configclass
class MultiRobotReachTerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


@configclass
class MultiRobotReachEventsCfg:
    """Reset events auto-dispatched across all robot groups.

    Both terms use ``per_robot=True`` — the manager auto-injects
    ``asset_cfg`` from each :class:`RobotSpec` and passes
    group-local ``env_ids``.
    """

    reset_to_default = EventTerm(
        func=mdp.reset_asset_to_default,
        mode="reset",
        per_robot=True,
        params={"reset_joint_targets": True},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        per_robot=True,
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
        },
    )


# ===========================================================
# Curriculum
# ===========================================================


@configclass
class MultiRobotReachCurriculumCfg:
    """Gradually increase action-rate and joint-vel penalties to suppress jitter."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 450},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 450},
    )


# ===========================================================
# Top-level env config
# ===========================================================


@configclass
class MultiRobotReachEnvCfg(ManagerBasedRLEnvCfg):
    """Multi-robot reach: 3 robots, shared 6D action columns.

    Group 0: OpenArm (7 arm DoF)
    Group 1: Franka  (7 arm DoF)
    Group 2: UR10    (6 arm DoF)

    Action dim: 6 (columns shared across disjoint groups).
    """

    scene: MultiRobotReachSceneCfg = MultiRobotReachSceneCfg(
        num_envs=2048,
        env_spacing=2.0,
        replicate_physics=False,
    )
    actions: MultiRobotReachActionsCfg = MultiRobotReachActionsCfg()
    commands: MultiRobotReachCommandsCfg = MultiRobotReachCommandsCfg()
    observations: MultiRobotReachObsCfg = MultiRobotReachObsCfg()
    rewards: MultiRobotReachRewardsCfg = MultiRobotReachRewardsCfg()
    terminations: MultiRobotReachTerminationsCfg = MultiRobotReachTerminationsCfg()
    events: MultiRobotReachEventsCfg = MultiRobotReachEventsCfg()
    curriculum: MultiRobotReachCurriculumCfg = MultiRobotReachCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 3
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()
