# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One goal-conditioned Franka policy sorts parcels across varied conveyor layouts."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import mdp as common_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import ResetStateSamplerCfg
from isaaclab_tasks.utils.sampling import BetaSamplingStrategyCfg, SamplerCfg, UniformSamplingStrategyCfg
from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

from ..conveyor_franka.conveyor_franka_env_cfg import ConveyorForceCfg, ConveyorFrankaEnvCfg
from ..conveyor_franka.mdp.rewards import transfer_success_reward
from ..conveyor_franka.mdp.terminations import invalid_action as invalid_policy_action
from . import mdp, observations
from .actions_cfg import TOOL_OFFSET, CartesianSuctionArmCfg, VacuumActionCfg
from .scene import PARCEL_SIZES, SuctionSortingSceneCfg, spawn_suction_franka

_ARM = [f"panda_joint{index}" for index in range(1, 8)]


@configclass
class ActionsCfg:
    arm_action = CartesianSuctionArmCfg(
        asset_name="robot",
        joint_names=_ARM,
        body_name="panda_hand",
        body_offset=CartesianSuctionArmCfg.OffsetCfg(pos=TOOL_OFFSET),
        scale=0.01,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )
    gripper_action = VacuumActionCfg()


@configclass
class CommandsCfg:
    sort = mdp.SizeSortCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joints = ObservationTermCfg(
            func=common_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM)}
        )
        joint_velocities = ObservationTermCfg(
            func=common_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM)}, scale=0.1
        )
        parcel = ObservationTermCfg(func=mdp.parcel_observation)
        tool = ObservationTermCfg(func=mdp.tool_observation)
        suction = ObservationTermCfg(func=mdp.suction_observation)
        goal = ObservationTermCfg(func=common_mdp.generated_commands, params={"command_name": "sort"})
        collision_point_cloud = ObservationTermCfg(
            func=observations.collision_point_cloud, noise=observations.CollisionPointCloudNoiseCfg()
        )
        actions = ObservationTermCfg(func=common_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    outfeed_layouts = EventTerm(func=mdp.initialize_outfeed_layouts, mode="prestartup")
    reset_parcels = EventTerm(
        func=mdp.ResetParcels,
        mode="reset",
        params={
            "sampling": ResetStateSamplerCfg(
                success_monitor=SuccessMonitorCfg(monitored_history_len=50),
                sampling=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.5, kappa=2.0),
                        UniformSamplingStrategyCfg(weight=0.0009),
                    ],
                    eps=1.0e-4,
                ),
            ),
            "full_starts": False,
            "samples_per_path": 64,
        },
    )


@configclass
class RewardsCfg:
    delivery = RewardTermCfg(func=transfer_success_reward, params={"command_name": "sort"}, weight=600.0)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=common_mdp.time_out, time_out=True)
    delivery_complete = TerminationTermCfg(func=mdp.delivery_complete)
    failure = TerminationTermCfg(func=mdp.failed_delivery)
    invalid_action = TerminationTermCfg(
        func=invalid_policy_action, params={"action_names": ("arm_action", "gripper_action")}
    )


@configclass
class SuctionSortingEnvCfg(ConveyorFrankaEnvCfg):
    class_type: type | str = "{DIR}.env:SuctionSortingEnv"
    scene: SuctionSortingSceneCfg = SuctionSortingSceneCfg(num_envs=512, env_spacing=8.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None
    episode_length_s: float = 120.0
    parcel_count: int = 5
    """Number of physical parcels instantiated in each environment, from 1 to 12."""
    conveyor_force: ConveyorForceCfg = ConveyorForceCfg(
        speed=0.25,
        friction=0.6,
        transported_body_pattern=r"(?:^|/)Cube[0-9]+(?:/|$)",
        transported_body_count_per_env=5,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Match moving-parcel resets at operating speed after full and partial resets.
        self.conveyor_force.startup_duration_s = self.sim.dt
        for index in range(len(PARCEL_SIZES)):
            asset_cfg = SceneEntityCfg(f"parcel_{index}")
            setattr(
                self.events,
                f"parcel_{index}_mass",
                EventTerm(
                    func=common_mdp.randomize_rigid_body_mass,
                    mode="startup",
                    params={
                        "asset_cfg": asset_cfg,
                        "mass_distribution_params": (0.7, 1.3),
                        "operation": "scale",
                        "recompute_inertia": True,
                        "min_mass": 0.01,
                    },
                ),
            )
            setattr(
                self.events,
                f"parcel_{index}_material",
                EventTerm(
                    func=common_mdp.randomize_rigid_body_material,
                    mode="startup",
                    params={
                        "asset_cfg": asset_cfg,
                        "static_friction_range": (0.4, 0.9),
                        "dynamic_friction_range": (0.4, 0.9),
                        "restitution_range": (0.0, 0.05),
                        "num_buckets": 1,
                    },
                ),
            )
        self.scene.robot.spawn.func = spawn_suction_franka
        self.scene.robot.spawn.variants = {"Physics": "mujoco", "Colliders": "primitives"}
        self.scene.robot.spawn.articulation_props = NewtonArticulationCfg(self_collision_enabled=False)
        self.scene.robot.actuators["panda_arm"].joint_velocity_limit = {
            "panda_joint[1-4]": 2.175,
            "panda_joint[5-7]": 2.61,
        }
        self.scene.arm_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*/panda_link[1-7]", update_period=self.sim.dt, history_length=2
        )
        self.scene.collision_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}",
            offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 1.1)),
            mesh_prim_paths=[],
            pattern_cfg=patterns.GridPatternCfg(resolution=0.24, size=(1.68, 1.68)),
            max_distance=2.0,
            update_period=self.sim.dt * self.decimation,
        )
        self.sim.physics.solver_cfg.nconmax = 400
        self.sim.physics.solver_cfg.njmax = 700
        self.sim.physics.collision_cfg.include_static_kinematic_pairs = False
        self.sim.visualizer_cfgs = VisualizerCfg(
            eye=(-2.6, -4.2, 2.8),
            lookat=(0.8, 0.2, -0.25),
            focal_length=32.0,
            max_visible_envs=1,
            randomly_sample_visible_envs=False,
        )

    def play_mode(self) -> None:
        self.scene.num_envs = min(self.scene.num_envs, 4)
        self.scene.outfeeds.spawn.assets_cfg = self.scene.outfeeds.spawn.assets_cfg[:1]
        self.events.reset_parcels.params["full_starts"] = True
        self.observations.policy.enable_corruption = False

    def validate_config(self) -> None:
        """Require the force-based suction and conveyor contact path."""
        if not isinstance(self.parcel_count, int) or not 1 <= self.parcel_count <= len(PARCEL_SIZES):
            raise ValueError(f"parcel_count must be an integer between 1 and {len(PARCEL_SIZES)}.")
        physics = self.sim.physics
        if not isinstance(physics, NewtonCfg) or not isinstance(physics.solver_cfg, MJWarpSolverCfg):
            raise ValueError("Suction sorting requires Newton MJWarp physics.")
        if physics.solver_cfg.use_mujoco_contacts or physics.collision_cfg is None:
            raise ValueError("Suction sorting requires Newton's external collision pipeline.")
