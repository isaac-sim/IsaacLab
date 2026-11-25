# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NIST composition with reset-selectable assembly variants."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.reset import ResetStateSamplerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .assembly_variants import ASSEMBLY_VARIANT_NAMES
from .nist_presets import EndEffectorBodyCfg, GripperGraspOffsetCfg, JointEffortNamesCfg
from .nist_reset_env_cfg import VARIANT_ACCUMULATOR_RESET
from .nist_scene_cfg import NISTSceneCfg
from .utils.variant_sampling import BalancedVariantScoreCfg


@configclass
class NISTObservationsCfg:
    """Policy state and scene geometry observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        held_asset_in_fixed_asset_frame = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            history_length=5,
            params={
                "target_asset_cfg": SceneEntityCfg("held_asset"),
                "root_asset_cfg": SceneEntityCfg("fixed_asset"),
                "target_asset_offset": "held_align",
                "root_asset_offset": "fixed_tip",
            },
        )
        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            history_length=5,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type: ignore
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset": GripperGraspOffsetCfg(),
            },
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=5)
        prev_action = ObsTerm(func=mdp.last_action, history_length=5)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PerceptionCfg(ObsGroup):
        scene_point_cloud = ObsTerm(
            func=mdp.scene_point_cloud_b,
            clip=(-2.0, 2.0),
            params={
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                "held_asset_cfg": SceneEntityCfg("held_asset"),
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "fixed_num_points": 256,
                "held_num_points": 256,
                "robot_num_points": 256,
                "flatten": True,
                "sampling_variant_order": None,
                "sampling_seed": 42,
                "robot_geometry_usd_path": None,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    perception: PerceptionCfg = PerceptionCfg()


@configclass
class NISTEventCfg:
    """Add assembly selection to the NIST startup events."""

    solver_fault_observations = EventTerm(func="isaaclab_newton.envs.mdp:zero_final_obs_on_solver_fault", mode="reset")
    assembly_variants = EventTerm(
        func=mdp.AssemblyVariantContext,
        mode="startup",
        params={
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "variant_names": ASSEMBLY_VARIANT_NAMES,
        },
    )
    held_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("held_asset"),
        },
    )
    fixed_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("fixed_asset"),
        },
    )
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_strategies = VARIANT_ACCUMULATOR_RESET


@configclass
class NISTRewardsCfg:
    """Reward assembly success and clear solver-fault rows."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_effort = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointEffortNamesCfg())},
        weight=-1e-4,
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)
    success_reward = RewTerm(func=mdp.success_reward, weight=25.0)
    solver_fault_reward = RewTerm(func="isaaclab_newton.envs.mdp:zero_reward_on_solver_fault", weight=1.0)


@configclass
class NISTTerminationsCfg:
    """Use variant geometry and Newton failure detection."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    assembly_contact_force = DoneTerm(
        func=mdp.assembly_contact_force,
        params={"threshold": 50.0, "sensor_cfg": SceneEntityCfg("assembly_contact")},
    )
    oob = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "in_bound_range": {"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
        },
    )
    progress_context = DoneTerm(
        func=mdp.variant_progress_context,
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
        },
    )
    abnormal = DoneTerm(
        func=mdp.joint_vel_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-7]")},
    )
    success = DoneTerm(func=mdp.success_termination)
    solver_fault = DoneTerm(func="isaaclab_newton.envs.mdp:solver_fault")


@configclass
class NISTActionsCfg:
    """Franka joint actions for the assembly variant task."""

    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale={"(?!panda_joint7$).*": 0.02, "panda_joint7": 0.2},
        use_zero_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class NISTEnvCfg(ManagerBasedRLEnvCfg):
    """NIST environment with reset-selectable assembly pairs."""

    class_type: type | str = "{DIR}.nist_env:NISTEnv"
    scene: NISTSceneCfg = NISTSceneCfg()
    observations: NISTObservationsCfg = NISTObservationsCfg()
    events: NISTEventCfg = NISTEventCfg()
    terminations: NISTTerminationsCfg = NISTTerminationsCfg()
    rewards: NISTRewardsCfg = NISTRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, 0.8, 0.4), lookat=(0.0, 0.0, 0.4))
    actions: NISTActionsCfg = NISTActionsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 14.0
        self.sim.dt = 0.04 / self.decimation
        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5
        self.sim.render_interval = self.decimation
        # Keep enough constraint and contact capacity for all 20 variants (including gear meshes)
        # while avoiding over-allocation and static/kinematic-vs-kinematic SDF contact overhead.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=1024,
                nconmax=256,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                ls_parallel=False,
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="sap",
                include_static_kinematic_pairs=False,
                max_triangle_pairs=90_000_000,
                rigid_contact_max=5_000_000,
                speculative_contact_gap_max=0.01,
                sdf_all_shapes=NewtonCollisionPipelineCfg.SDFAllShapesCfg(
                    sdf_max_resolution=256,
                    sdf_narrow_band_inner=-0.005,
                    sdf_narrow_band_outer=0.005,
                ),
            ),
            num_substeps=8,
            debug_mode=False,
            use_cuda_graph=True,
        )

    def play_mode(self) -> None:
        """Evaluate from uniformly sampled random-start states."""
        self.scene.num_envs = 128
        uniform = ResetStateSamplerCfg(
            history_length=1, scores=[BalancedVariantScoreCfg(beta_weight=0.0, floor=1.0)], uniform_weight=0.0
        )
        reset = self.events.reset_strategies.params
        reset["state_table_size"] = 500
        reset["sampling"] = uniform
        scene_reset = reset.get("reset_term", self.events.reset_strategies)
        choice = scene_reset.params["terms"]["reset_strategies"].params
        choice["terms"] = {"start_random": choice["terms"]["start_random"]}
