# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import config_field

from isaaclab_tasks.contrib.nist import mdp
from isaaclab_tasks.contrib.nist.factory_presets import (
    AssemblyTipCfg,
    FactoryAssemblyProfileCfg,
    HeldAssetAlignOffsetCfg,
)
from isaaclab_tasks.contrib.nist.factory_scenes_cfg import FactorySceneCfg
from isaaclab_tasks.contrib.nist.reset_env_cfg import ACCUMULATOR_RESET
from isaaclab_tasks.contrib.nist.utils import SamplerCfg, UniformSamplingStrategyCfg
from isaaclab_tasks.utils import PresetCfg, preset

_FRANKA_END_EFFECTOR = "panda_fingertip_centered"


@dataclass
class FactoryObservationsCfg:
    """Observation specifications for Factory."""

    @dataclass
    class PolicyCfg(ObsGroup):
        end_effector_vel_lin_ang_b: Any = config_field(
            ObsTerm(
                func=mdp.asset_link_velocity_in_root_asset_frame,
                params={
                    "target_asset_cfg": SceneEntityCfg("robot", body_names=_FRANKA_END_EFFECTOR),
                    "root_asset_cfg": SceneEntityCfg("robot"),
                },
            )
        )

        end_effector_pose: Any = config_field(
            ObsTerm(
                func=mdp.target_asset_pose_in_root_asset_frame,
                params={
                    "target_asset_cfg": SceneEntityCfg("robot", body_names=_FRANKA_END_EFFECTOR),
                    "root_asset_cfg": SceneEntityCfg("robot"),
                    "target_asset_offset": AssemblyTipCfg(),
                },
            )
        )

        held_asset_in_fixed_asset_frame: ObsTerm = config_field(
            ObsTerm(
                func=mdp.target_asset_pose_in_root_asset_frame,
                params={
                    "target_asset_cfg": SceneEntityCfg("held_asset"),
                    "root_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "root_asset_offset": AssemblyTipCfg(),
                },
            )
        )

        fixed_asset_in_end_effector_frame: ObsTerm = config_field(
            ObsTerm(
                func=mdp.target_asset_pose_in_root_asset_frame,
                params={
                    "target_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "root_asset_cfg": SceneEntityCfg("robot", body_names=_FRANKA_END_EFFECTOR),
                    "target_asset_offset": AssemblyTipCfg(),
                },
            )
        )

        joint_pos: Any = config_field(ObsTerm(func=mdp.joint_pos))

        prev_action: Any = config_field(ObsTerm(func=mdp.last_action))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = config_field(PolicyCfg())
    critic: PolicyCfg = config_field(PolicyCfg())


@dataclass
class FactoryEventCfg:
    """Events specifications for Factory"""

    held_asset_material: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_material,  # type: ignore
            mode="startup",
            params={
                "static_friction_range": (0.4, 1.0),
                "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "asset_cfg": SceneEntityCfg("held_asset"),
            },
        )
    )

    # Increase diagonal inertia to prevent contact-induced angular instability.
    held_asset_inertia: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("held_asset"),
                "inertia_distribution_params": [0.001, 0.001],
                "operation": "add",
                "diagonal_only": True,
            },
        )
    )

    fixed_asset_material: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_material,  # type: ignore
            mode="startup",
            params={
                "static_friction_range": (0.4, 1.0),
                "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )
    )

    robot_material: Any = config_field(
        EventTerm(
            func=mdp.randomize_rigid_body_material,  # type: ignore
            mode="startup",
            params={
                "static_friction_range": (0.75, 0.75),
                "dynamic_friction_range": (0.75, 0.75),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    )

    reset_strategies: Any = config_field(ACCUMULATOR_RESET)

    # The curriculum restores gravity as task difficulty rises.
    variable_gravity: EventTerm | None = config_field(
        EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="reset",
            params={"operation": "abs", "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))},
        )
    )


@dataclass
class FactoryRewardsCfg:
    """Reward terms for Factory. Success is terminal and carries the dominant weight."""

    action_l2: Any = config_field(RewTerm(func=mdp.action_l2_clamped, weight=-1e-4))
    action_rate_l2: Any = config_field(RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4))
    joint_effort: Any = config_field(
        RewTerm(
            func=mdp.joint_torques_l2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!panda_joint7$|panda_finger_.*$).*")},
            weight=-1e-4,
        )
    )
    early_termination: Any = config_field(
        RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)
    )
    success_reward: Any = config_field(RewTerm(func=mdp.success_reward, weight=100.0))


@dataclass
class FactoryTerminationsCfg:
    """Termination terms for Factory. Reaching the assembled pose ends the episode."""

    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))

    oob: Any = config_field(
        DoneTerm(
            func=mdp.out_of_bound,
            params={
                "asset_cfg": SceneEntityCfg("held_asset"),
                "in_bound_range": {"x": (-0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
            },
        )
    )

    progress_context: Any = config_field(
        DoneTerm(
            func=mdp.progress_context,
            params={
                "success_threshold": 0.001,
                "held_asset_cfg": SceneEntityCfg("held_asset"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                "held_asset_offset": HeldAssetAlignOffsetCfg(),
                "assembly_profile": FactoryAssemblyProfileCfg(),
            },
        )
    )

    abnormal: Any = config_field(
        DoneTerm(
            func=mdp.joint_vel_out_of_limit,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-6]")},
        )
    )

    wrist_limit: Any = config_field(
        DoneTerm(
            func=mdp.joint_vel_out_of_manual_limit,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint7"), "max_velocity": 8.0},
        )
    )

    success: Any = config_field(DoneTerm(func=mdp.success_termination))


@dataclass
class FactoryCurriculumsCfg:
    """Curriculum terms for Factory."""

    difficulty_scheduler: Any = config_field(
        CurrTerm(
            func=mdp.DifficultyScheduler,
            params={
                "max_difficulty": 10,
                "success_rate_callback": preset(
                    default="env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
                    accumulator="env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
                    choice="env.event_manager.get_term_cfg('reset_strategies').func.terms['reset_strategies'].func.term_success_rate",
                ),
            },
        )
    )

    gravity_adr: CurrTerm | None = config_field(
        CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "events.variable_gravity.params.gravity_distribution_params",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                    "difficulty_term_str": "difficulty_scheduler",
                },
            },
        )
    )


##
# Environment configuration
##


@dataclass
class FactoryPhysicsCfg(PresetCfg):
    """Factory physics backend presets."""

    isaacsim_physx: Any = config_field(
        PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**32 - 1,
            gpu_max_num_partitions=1,
            gpu_found_lost_pairs_capacity=2**22,
        )
    )
    newton_mjwarp: Any = config_field(
        NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=1500,
                nconmax=400,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                ls_parallel=False,
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="sap",
                max_triangle_pairs=60_000_000,
                rigid_contact_max=5_000_000,
            ),
            num_substeps=16,
            debug_mode=False,
            use_cuda_graph=True,
        )
    )
    physx: Any = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx))
    default: Any = config_field(isaacsim_physx)


@dataclass
class FactoryActionsCfg:
    """Franka joint actions for Factory."""

    arm_action: Any = config_field(
        mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale={"(?!panda_joint7$).*": 0.02, "panda_joint7": 0.2},
            use_zero_offset=True,
        )
    )
    gripper_action: Any = config_field(
        mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
    )


@dataclass
class FactoryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka Factory environment."""

    scene: FactorySceneCfg = config_field(FactorySceneCfg())
    observations: FactoryObservationsCfg = config_field(FactoryObservationsCfg())
    events: FactoryEventCfg = config_field(FactoryEventCfg())
    terminations: FactoryTerminationsCfg = config_field(FactoryTerminationsCfg())
    rewards: FactoryRewardsCfg = config_field(FactoryRewardsCfg())
    curriculum: FactoryCurriculumsCfg = config_field(FactoryCurriculumsCfg())
    viewer: ViewerCfg = config_field(ViewerCfg(eye=(0.0, 0.8, 0.4), lookat=(0.0, 0.0, 0.4)))
    actions: FactoryActionsCfg = config_field(FactoryActionsCfg())

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 8
        self.episode_length_s = 14.0
        # simulation settings
        self.sim.dt = 0.04 / self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics = FactoryPhysicsCfg()

        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5

    def play_mode(self) -> None:
        """Narrow the reset curriculum for evaluation.

        Training samples a curriculum over several reset strategies and a large bank of
        stored states; a policy is instead scored on one strategy, drawn uniformly, so the
        number reflects the task rather than whatever the curriculum currently favors.

        Called by :func:`~isaaclab_tasks.utils.hydra.register_task` after preset resolution,
        so it edits the already-resolved terms.
        """
        # Training starts at zero gravity and lets the ADR curriculum ramp it to -9.81 as the
        # difficulty rises. Evaluation is scored at full gravity, so both terms come off and the
        # sim keeps its configured value.
        self.events.variable_gravity = None
        self.curriculum.gravity_adr = None

        # ``play`` reads the training default when ``--num_envs`` is omitted, and the training
        # default is sized for throughput, not for watching a policy.
        self.scene.num_envs = 128

        uniform = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)

        reset = self.events.reset_strategies.params

        if "state_table_size" in reset:
            reset["state_table_size"] = 512
        if "sampling" in reset:
            reset["sampling"] = uniform

        scene_reset = reset.get("reset_term", self.events.reset_strategies)
        choice = scene_reset.params["terms"]["reset_strategies"].params
        choice["terms"] = {"grasp_asset_in_air": choice["terms"]["grasp_asset_in_air"]}
        choice["sampling"] = uniform
