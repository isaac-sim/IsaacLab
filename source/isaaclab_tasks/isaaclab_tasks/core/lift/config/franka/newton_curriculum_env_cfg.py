# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-only curriculum configuration for Franka cube lifting."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.lift_env_cfg import LiftPhysicsCfg

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class NewtonCurriculumObservationsCfg:
    """Task-relative observations for the Newton lift policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        ee_to_object = ObsTerm(func=mdp.object_position_relative_to_ee)
        object_to_goal = ObsTerm(
            func=mdp.object_goal_position_relative,
            params={"command_name": "object_pose"},
        )
        object_orientation_to_goal = ObsTerm(
            func=mdp.object_goal_orientation_error,
            params={"command_name": "object_pose"},
        )
        object_height = ObsTerm(func=mdp.object_height_above_table)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class NewtonCurriculumEventsCfg:
    """Reset events for the adaptive grasp-to-lift curriculum."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_curriculum = EventTerm(
        func=mdp.reset_franka_lift_curriculum,
        mode="reset",
        params={"closed_finger_position": 0.016},
    )


@configclass
class NewtonCurriculumRewardsCfg:
    """Sparse progress shaping with a dominant terminal success bonus."""

    reaching_object = RewTerm(func=mdp.curriculum_object_ee_distance, params={"std": 0.08}, weight=2.0)
    lift_progress = RewTerm(
        func=mdp.object_lift_progress,
        params={"minimal_height": 0.02, "target_height": 0.20},
        weight=1.0,
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.12, "minimal_height": 0.10, "command_name": "object_pose"},
        weight=3.0,
    )
    object_goal_orientation = RewTerm(
        func=mdp.object_goal_orientation_distance,
        params={"std": 0.25, "minimal_height": 0.10, "command_name": "object_pose"},
        weight=3.0,
    )
    object_goal_fine_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.025, "minimal_height": 0.10, "command_name": "object_pose"},
        weight=2.0,
    )
    object_goal_fine_orientation = RewTerm(
        func=mdp.object_goal_orientation_distance,
        params={"std": 0.10, "minimal_height": 0.10, "command_name": "object_pose"},
        weight=2.0,
    )
    object_goal_pose_accuracy = RewTerm(
        func=mdp.object_goal_pose_accuracy,
        params={
            "position_threshold": 0.02,
            "orientation_threshold": 0.15,
            "command_name": "object_pose",
        },
        weight=10.0,
    )
    success_bonus = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": "success"},
        # RewardManager multiplies by the 0.02-s control step: 5000 * 0.02 = 100.
        weight=5000.0,
    )
    close_near_object = RewTerm(
        func=mdp.curriculum_gripper_close_near_object,
        params={"std": 0.05},
        weight=2.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    object_dropping = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50.0,
        params={"term_keys": "object_dropping"},
    )


@configclass
class NewtonCurriculumTerminationsCfg:
    """Terminate on success, timeout, or an irrecoverable object drop."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.ObjectPoseHeld,
        params={
            "command_name": "object_pose",
            "position_threshold": 0.02,
            "orientation_threshold": 0.15,
            "hold_time": 1.0,
        },
    )
    object_dropping = DoneTerm(
        func=mdp.curriculum_object_below_reset_height,
        params={
            "high_object_height": 0.349140,
            "low_object_height": 0.029296,
            "transition_start": 0.30,
            "transition_end": 0.55,
            "height_margin": 0.10,
            "minimum_height": -0.05,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class NewtonLiftCurriculumCfg:
    """Per-environment mastery curriculum for Newton cube lifting."""

    lift_difficulty = CurrTerm(
        func=mdp.LiftDifficultyScheduler,
        params={
            "success_termination_name": "success",
            "initial_difficulty": 0,
            "max_difficulty": 40,
            "successes_to_promote": 1,
        },
    )


@configclass
class FrankaCubeLiftNewtonCurriculumEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka lift task whose physics and learning curriculum are Newton-only."""

    def __post_init__(self):
        super().__post_init__()

        # Pin this task to Newton so it cannot silently fall back to PhysX.
        self.sim.physics = LiftPhysicsCfg().newton_mjwarp

        # Newton has no gravity compensation; use the existing high-PD, gravity-free
        # Franka model so reset IK configurations remain controllable.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["panda_shoulder"].armature = 0.1
        self.scene.robot.actuators["panda_forearm"].armature = 0.1
        self.scene.robot.actuators["panda_shoulder"].stiffness = 20000.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 20000.0
        self.scene.robot.actuators["panda_shoulder"].damping = 2000.0
        self.scene.robot.actuators["panda_forearm"].damping = 2000.0
        self.scene.robot.actuators["panda_shoulder"].effort_limit_sim = 200.0
        self.scene.robot.actuators["panda_forearm"].effort_limit_sim = 100.0
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 40.0

        # Relative task-space control lets the policy reuse the same approach and
        # transport motion across the curriculum's changing reset poses.
        self.actions.arm_action = mdp.CurriculumDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
                joint_limit_avoidance_gain=0.1,
            ),
            scale=(0.1, 0.1, 0.1, 0.25, 0.25, 0.25),
            body_offset=mdp.CurriculumDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            full_control_difficulty=0.30,
        )
        self.actions.gripper_action = mdp.CurriculumGripperActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.016},
            force_close_below_difficulty=0.45,
        )

        self.commands.object_pose = mdp.CurriculumPoseCommandCfg(
            asset_name="robot",
            body_name="panda_hand",
            resampling_time_range=(5.0, 5.0),
            debug_vis=False,
            tracked_object_name="object",
            easy_goal=(0.499620, 0.0, 0.349140),
            full_goal_difficulty=0.30,
            ranges=mdp.CurriculumPoseCommandCfg.Ranges(
                pos_x=(0.4, 0.6),
                pos_y=(-0.25, 0.25),
                pos_z=(0.25, 0.5),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0.0, 0.0),
            ),
        )
        self.observations = NewtonCurriculumObservationsCfg()
        self.events = NewtonCurriculumEventsCfg()
        self.rewards = NewtonCurriculumRewardsCfg()
        self.terminations = NewtonCurriculumTerminationsCfg()
        self.curriculum = NewtonLiftCurriculumCfg()


@configclass
class FrankaCubeLiftNewtonCurriculumEnvCfg_PLAY(FrankaCubeLiftNewtonCurriculumEnvCfg):
    """Evaluation configuration that starts every environment at final difficulty."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.curriculum.lift_difficulty.params["initial_difficulty"] = 40
        self.commands.object_pose.debug_vis = False
