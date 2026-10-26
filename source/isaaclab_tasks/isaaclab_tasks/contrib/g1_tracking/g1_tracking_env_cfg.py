# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""The G1 tracking task's native scene, manager terms and Newton physics preset."""

from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    CommandTermCfg,
    DatasetExportMode,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderManagerBaseCfg,
    RecorderTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.utils import PresetCfg

from .motion import CONTROL_DT
from .tracking import RewardCfg

G1_DATA_DIR = Path(ISAACLAB_ROOT) / "data/g1_tracking"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


# Per-joint (effort limit, velocity limit, armature) by motor type.
_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


G1_CFG = ArticulationCfg(
    # Prepare the matching 29-DoF MJCF and convert it to USD before constructing this task.
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(G1_DATA_DIR / "g1_usd/g1_29dof/g1_29dof.usda"),
        variants={"Physics": "mujoco"},
        activate_contact_sensors=False,
        rigid_props=PhysxRigidBodyCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
        articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            joint_effort_limit={f".*{name}_joint": motor[0] for name, motor in JOINT_MOTORS.items()},
            joint_velocity_limit={f".*{name}_joint": motor[1] for name, motor in JOINT_MOTORS.items()},
            stiffness={f".*{name}_joint": motor[2] * NATURAL_FREQ**2 for name, motor in JOINT_MOTORS.items()},
            damping={
                f".*{name}_joint": 2 * DAMPING_RATIO * motor[2] * NATURAL_FREQ for name, motor in JOINT_MOTORS.items()
            },
            armature={f".*{name}_joint": motor[2] for name, motor in JOINT_MOTORS.items()},
        )
    },
)


@configclass
class TrackingPhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True
        ),
        num_substeps=4,
        use_cuda_graph=True,
    )
    default = newton_mjwarp


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.3, dynamic_friction=0.3)
        ),
    )
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type = "isaaclab_tasks.contrib.g1_tracking.mdp:MotionCommand"
    resampling_time_range = (1.0e9, 1.0e9)
    motion_dir: str = str(G1_DATA_DIR / "lafan1_g1")
    """Directory containing metric/xyzw/radian 36-channel motion CSV or NumPy files."""
    kinematics_path: str = str(G1_DATA_DIR / "g1_description/g1_29dof.xml")
    """MJCF matching the scene asset, used for differentiable link kinematics."""
    source_fps: float = 30.0
    goal_frames: int = 10
    gamma: float = 0.99
    evaluation: bool = False
    evaluation_clip_offset: int = 0
    reset_noise: bool = True
    objective: RewardCfg = RewardCfg()


@configclass
class CommandsCfg:
    motion = MotionCommandCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        class_type="isaaclab_tasks.contrib.g1_tracking.mdp:ContextJointPositionAction",
        asset_name="robot",
        joint_names=[".*"],
        scale={
            name: 0.25 * G1_CFG.actuators["joints"].joint_effort_limit[name] / stiffness
            for name, stiffness in G1_CFG.actuators["joints"].stiffness.items()
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(
            func="isaaclab_tasks.contrib.g1_tracking.mdp:joint_pos_rel", noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5))
        angular_velocity = ObservationTermCfg(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
        previous_action = ObservationTermCfg(func=mdp.last_action)
        orientation = ObservationTermCfg(
            func="isaaclab_tasks.contrib.g1_tracking.mdp:relative_orientation",
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        history_length = 10
        flatten_history_dim = False
        concatenate_terms = True
        enable_corruption = True

    @configclass
    class GoalCfg(ObservationGroupCfg):
        reference = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"})

    policy = PolicyCfg()
    goal = GoalCfg()


@configclass
class RewardsCfg:
    tracking = RewardTermCfg(func="isaaclab_tasks.contrib.g1_tracking.mdp:tracking_reward", weight=1.0)


@configclass
class TerminationsCfg:
    failed = TerminationTermCfg(func="isaaclab_tasks.contrib.g1_tracking.mdp:tracking_failure")
    clip_end = TerminationTermCfg(func="isaaclab_tasks.contrib.g1_tracking.mdp:clip_end", time_out=True)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    context = EventTermCfg(func="isaaclab_tasks.contrib.g1_tracking.mdp:randomize_context", mode="startup")


@configclass
class TransitionRecorderCfg(RecorderTermCfg):
    class_type = "isaaclab_tasks.contrib.g1_tracking.mdp:TransitionRecorder"
    record_to_episodes: bool = False


@configclass
class RecordersCfg(RecorderManagerBaseCfg):
    transition: TransitionRecorderCfg | None = None
    dataset_export_mode = DatasetExportMode.EXPORT_NONE


@configclass
class G1TrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene = TrackingSceneCfg(num_envs=4096, env_spacing=0.0)
    commands = CommandsCfg()
    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    recorders = RecordersCfg()
    sim = sim_utils.SimulationCfg(dt=CONTROL_DT, render_interval=1, physics=TrackingPhysicsCfg())
    decimation = 1
    episode_length_s = 10.0

    def play_mode(self):
        self.commands.motion.evaluation = True
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 1.0e9
        self.scene.ground.spawn.physics_material.static_friction = 1.0
        self.scene.ground.spawn.physics_material.dynamic_friction = 1.0
