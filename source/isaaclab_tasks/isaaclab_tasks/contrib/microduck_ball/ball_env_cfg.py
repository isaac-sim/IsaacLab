# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck balancing and rolling on a ball, in four stages.

1. :class:`MicroDuckBallFixedEnvCfg`: the ball cannot move.
2. :class:`MicroDuckBallBalanceEnvCfg`: the ball rolls freely and the robot keeps it under its feet.
3. :class:`MicroDuckBallRollEnvCfg`: a velocity command asks the robot to step on the ball and roll it.
4. :class:`MicroDuckBallRobustEnvCfg`: pushes, randomized friction and ball mass, and servo latency.

The stages share observations and actions, so the checkpoint of one stage can initialize the next.
"""

import math

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.microduck import MICRODUCK_CFG

from . import mdp

BALL_RADIUS = 0.12
"""Radius of the ball [m]. A flat sole on a sphere is stable on its own only if the center of mass is below it."""
BALL_MASS = 0.3
"""Mass of the ball [kg]."""
STANDING_HEIGHT = 0.0905
"""Height of the base above the soles in the crouched default pose [m]."""
FOOT_CONTACT_HEIGHT = 0.04
"""Distance of a foot from the ball surface below which it counts as standing on the ball [m]."""
FOOT_REST_HEIGHT = 0.031
"""Distance of a foot standing on the ball from its surface [m]."""
SWING_CLEARANCE = 0.035
"""Height of a step above the resting foot [m]."""

_FOOT_CFG = SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"])
_BALL_CFG = SceneEntityCfg("ball")
_SERVO_CFG = {
    "joint_names_expr": [".*"],
    "actuator_effort_limit": 0.96,
    "actuator_velocity_limit": 12.0,
    "stiffness": 5.0,
    "damping": 0.1,
    "armature": 0.0018,
    "friction": 0.0048,
}
"""Position servo: a PD law followed by the torque limit of the Dynamixel XL330."""

MICRODUCK_BALL_CFG = MICRODUCK_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=MICRODUCK_CFG.spawn.replace(
        rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=1.0),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
    ),
    # crouched: knees bent, hip and ankle chosen so that the soles are flat and the center of mass is over them
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2 * BALL_RADIUS + STANDING_HEIGHT + 0.003),
        joint_pos={
            ".*hip_yaw": 0.0,
            "left_hip_roll": -0.0873,
            "right_hip_roll": 0.0873,
            "left_hip_pitch": 0.2043,
            "right_hip_pitch": -0.2043,
            "left_knee": 0.9,
            "right_knee": -0.9,
            "left_ankle": 0.6957,
            "right_ankle": -0.6957,
            "neck_pitch": 0.3491,
            "head_pitch": 0.3491,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={"servos": IdealPDActuatorCfg(**_SERVO_CFG)},
)
"""MicroDuck with ideal position servos, in the crouched pose that is the default pose of the policy."""


@configclass
class MicroDuckBallSceneCfg(InteractiveSceneCfg):
    """Scene with the MicroDuck and a ball on a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MICRODUCK_BALL_CFG
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            mass_props=sim_utils.MassCfg(mass=BALL_MASS),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            collision_props=PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.25, 0.02), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_RADIUS)),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.BallVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        debug_vis=False,
        ranges=mdp.BallVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # the neck and head stay at their default pose; the hips barely yaw
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_hip_yaw": 0.1,
            ".*_hip_roll": 0.25,
            ".*_(hip_pitch|knee|ankle)": 0.5,
            "neck_pitch": 0.0,
            "head_.*": 0.0,
        },
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group. The ball is read from the simulator, a real robot would estimate it."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        gait_clock = ObsTerm(func=mdp.gait_clock, params={"command_name": "base_velocity"})
        ball_pos = ObsTerm(
            func=mdp.ball_pos_b,
            params={"robot_cfg": _FOOT_CFG, "ball_cfg": _BALL_CFG},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        ball_vel = ObsTerm(
            func=mdp.ball_vel_b,
            params={"robot_cfg": _FOOT_CFG, "ball_cfg": _BALL_CFG},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Additional observations of the critic."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ball_ang_vel = ObsTerm(func=mdp.ball_ang_vel_w, params={"ball_cfg": _BALL_CFG})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Event specifications for the MDP. The randomization ranges are widened by the robust stage."""

    feet_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": _FOOT_CFG,
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": _BALL_CFG,
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": _BALL_CFG, "mass_distribution_params": (1.0, 1.0), "operation": "scale"},
    )
    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": _BALL_CFG},
    )
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    push_robot: EventTerm | None = None
    push_ball: EventTerm | None = None


@configclass
class RewardsCfg:
    """Reward terms for the MDP. The stepping and tracking terms are switched on by the velocity stage."""

    upright = RewTerm(func=mdp.upright, weight=1.0, params={"std": 0.25})
    ball_under_feet = RewTerm(
        func=mdp.ball_under_feet, weight=1.0, params={"std": 0.05, "robot_cfg": _FOOT_CFG, "ball_cfg": _BALL_CFG}
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.2}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_lin_vel_xy_linear = RewTerm(
        func=mdp.track_lin_vel_xy_linear, weight=0.0, params={"command_name": "base_velocity", "scale": 0.3}
    )
    track_ang_vel_z_linear = RewTerm(
        func=mdp.track_ang_vel_z_linear, weight=0.0, params={"command_name": "base_velocity", "scale": 1.0}
    )
    lean_into_motion = RewTerm(
        func=mdp.lean_into_motion,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.08,
            "lean_per_speed": 0.5,
            "lean_per_error": 1.0,
            "max_lean": 0.15,
        },
    )
    gait_clock_tracking = RewTerm(
        func=mdp.gait_clock_tracking,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.012,
            "clearance": SWING_CLEARANCE,
            "rest_height": FOOT_REST_HEIGHT,
            "robot_cfg": _FOOT_CFG,
            "ball_cfg": _BALL_CFG,
        },
    )
    ball_slip = RewTerm(func=mdp.ball_slip_l2, weight=-0.5, params={"robot_cfg": _FOOT_CFG, "ball_cfg": _BALL_CFG})
    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=0.0,
        params={"contact_height": FOOT_CONTACT_HEIGHT, "robot_cfg": _FOOT_CFG, "ball_cfg": _BALL_CFG},
    )
    idle_feet_lift = RewTerm(
        func=mdp.idle_feet_lift,
        weight=0.0,
        params={"command_name": "base_velocity", "robot_cfg": _FOOT_CFG, "ball_cfg": _BALL_CFG},
    )
    heading_error = RewTerm(func=mdp.heading_error_l1, weight=0.0, params={"command_name": "base_velocity"})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    terminated = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    duck_tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    duck_too_low = DoneTerm(
        func=mdp.base_too_low,
        params={"minimum_height": 0.05, "robot_cfg": SceneEntityCfg("robot"), "ball_cfg": _BALL_CFG},
    )
    ball_off_feet: DoneTerm | None = DoneTerm(
        func=mdp.ball_off_feet, params={"max_distance": 0.07, "robot_cfg": _FOOT_CFG, "ball_cfg": _BALL_CFG}
    )
    ball_outside_zone: DoneTerm | None = DoneTerm(
        func=mdp.ball_outside_zone, params={"radius": 0.6, "ball_cfg": _BALL_CFG}
    )


@configclass
class MicroDuckBallBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Free ball, no velocity command: stay upright and keep the ball under the feet."""

    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    scene: MicroDuckBallSceneCfg = MicroDuckBallSceneCfg(num_envs=2048, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.7, 0.7, 0.4), lookat=(0.0, 0.0, 0.2))

    def play_mode(self):
        """Use the shared inference defaults, a longer episode and no pushes."""
        super().play_mode()
        self.episode_length_s = 20.0
        self.events.push_robot = None
        self.events.push_ball = None


@configclass
class MicroDuckBallFixedEnvCfg(MicroDuckBallBalanceEnvCfg):
    """Fixed ball: the robot stands on a ball that cannot move."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.ball.spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)
        self.rewards.ball_under_feet.weight = 0.0
        self.rewards.ball_slip.weight = 0.0
        self.terminations.ball_off_feet = None
        self.terminations.ball_outside_zone = None


@configclass
class MicroDuckBallRollEnvCfg(MicroDuckBallBalanceEnvCfg):
    """Velocity command: step on the ball and roll it where the command says, whichever way the ball is pushed."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
        self.terminations.ball_outside_zone = None
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.12, 0.12)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.push_ball = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, "asset_cfg": _BALL_CFG},
        )
        self.rewards.upright.weight = 0.5
        self.rewards.upright.params["std"] = 0.4
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.08
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.track_lin_vel_xy_linear.weight = 1.0
        self.rewards.track_ang_vel_z_linear.weight = 1.0
        self.rewards.lean_into_motion.weight = 1.0
        self.rewards.gait_clock_tracking.weight = 2.0
        self.rewards.feet_slip.weight = -3.0
        self.rewards.idle_feet_lift.weight = -6.0
        self.rewards.heading_error.weight = -1.5


@configclass
class MicroDuckBallRobustEnvCfg(MicroDuckBallRollEnvCfg):
    """Velocity command with pushes, randomized friction and ball mass, and 0 to 2 physics steps of servo latency."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["servos"] = DelayedPDActuatorCfg(**_SERVO_CFG, min_delay=0, max_delay=2)
        self.events.feet_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.feet_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.events.ball_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.ball_material.params["dynamic_friction_range"] = (0.4, 1.0)
        self.events.ball_mass.params["mass_distribution_params"] = (0.6, 1.6)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
        )
        self.events.push_ball.params["velocity_range"] = {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
