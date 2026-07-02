# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
from pathlib import Path

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.deploy.mdp.delayed_joint_actions_cfg import ShapedDelayedRelativeJointPositionActionCfg

from .joint_pos_env_cfg import Rizon4sGearAssemblyEnvCfg

ISAACLAB_ROOT = Path(__file__).resolve().parents[8]
FLEXIV_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
FLEXIV_ACTION_LATENCY_MS = 20.0
FLEXIV_PHYSX_SYSID_ACTUATOR_YAML = "flexiv/manual/flexiv_pd_only_gravityoff_high_cmdlimits_tuned_physx.yaml"
FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH = f"input/actuator_models/{FLEXIV_PHYSX_SYSID_ACTUATOR_YAML}"
FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ = 200.0
FLEXIV_DEPLOYMENT_DECIMATION = 4
FLEXIV_DEPLOYMENT_CONTROL_FREQ_HZ = FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ / FLEXIV_DEPLOYMENT_DECIMATION
FLEXIV_ROBOT_COLLECTION_COMMAND_VELOCITY_LIMIT = 2.0
FLEXIV_ROBOT_COLLECTION_COMMAND_ACCELERATION_LIMIT = 3.0
FLEXIV_GRAV_ROS_INFERENCE_SETUP = {
    "robot_usd": "rizon4s_with_grav.usd",
    "gravity_compensation": "robot_rigid_bodies_disable_gravity_true",
    "rigid_body": {
        "disable_gravity": True,
        "max_depenetration_velocity": 5.0,
        "linear_damping": 0.0,
        "angular_damping": 0.0,
        "max_linear_velocity": 1000.0,
        "max_angular_velocity": 3666.0,
        "enable_gyroscopic_forces": True,
        "solver_position_iteration_count": 4,
        "solver_velocity_iteration_count": 1,
        "max_contact_impulse": 1e32,
    },
    "articulation": {
        "enabled_self_collisions": False,
        "solver_position_iteration_count": 4,
        "solver_velocity_iteration_count": 1,
    },
    "collision": {"contact_offset": 0.005, "rest_offset": 0.0},
    "arm_actuator_yaml": FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH,
    "action_latency_ms": FLEXIV_ACTION_LATENCY_MS,
}


def constant_obs(env, value: tuple) -> torch.Tensor:
    """Observation function that returns a fixed tensor every step."""
    return torch.tensor([value], device=env.device, dtype=torch.float32).expand(env.num_envs, -1)


def _load_implicit_actuator_cfg(actuator_yaml: str):
    if str(ISAACLAB_ROOT) not in sys.path:
        sys.path.append(str(ISAACLAB_ROOT))
    from input.actuator_models import load_implicit_actuator_cfg

    return load_implicit_actuator_cfg(actuator_yaml, FLEXIV_ARM_JOINT_NAMES)


def _replace_arm_actuator(robot_cfg, actuator_yaml: str) -> None:
    """Install a single tuned arm actuator while preserving gripper actuators."""
    gripper_actuators = {
        actuator_name: actuator_cfg
        for actuator_name, actuator_cfg in robot_cfg.actuators.items()
        if actuator_name.startswith("gripper")
    }
    robot_cfg.actuators = {
        "arm": _load_implicit_actuator_cfg(actuator_yaml),
        **gripper_actuators,
    }


@configclass
class Rizon4sGearAssemblyROSInferenceEnvCfg(Rizon4sGearAssemblyEnvCfg):
    """Configuration for ROS inference with Flexiv Rizon 4s and Grav gripper.

    This configuration:
    - Exposes variables needed for ROS inference
    - Overrides robot and gear initial poses for fixed/deterministic setup
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Variables used by Isaac Manipulator for on robot inference
        # These parameters allow the ROS inference node to validate environment configuration,
        # perform checks during inference, and correctly interpret observations and actions.
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "shaft_pos", "shaft_quat"]
        self.policy_action_space = "joint"
        # Use inherited joint names from parent's observation configuration
        self.arm_joint_names = self.observations.policy.joint_pos.params["asset_cfg"].joint_names
        # Use inherited num_arm_joints from parent
        self.action_space = self.num_arm_joints
        # State space and observation space for Rizon 4s with Grav gripper (7 DOF arm + 1 gripper)
        # State: 7 joint pos + 7 joint vel + 3 shaft pos + 4 shaft quat + 3 gear pos + 4 gear quat = 28
        # For critic: additional gear observations
        self.state_space = 28
        # Observation: 7 joint pos + 7 joint vel + 3 shaft pos + 4 shaft quat = 21
        self.observation_space = 21

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        # Dynamically generate action_scale_joint_space based on action_space
        self.action_scale_joint_space = [self.joint_action_scale] * self.action_space

        _replace_arm_actuator(self.scene.robot, FLEXIV_PHYSX_SYSID_ACTUATOR_YAML)
        self.decimation = FLEXIV_DEPLOYMENT_DECIMATION
        self.sim.dt = 1.0 / FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ
        self.sim.render_interval = self.decimation
        self.flexiv_tuned_actuator_yaml = FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH
        self.flexiv_action_latency_ms = FLEXIV_ACTION_LATENCY_MS
        self.sim_to_real_tuned_config = dict(FLEXIV_GRAV_ROS_INFERENCE_SETUP)
        self.sim_to_real_tuned_config.update(
            {
                "active_env": "IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference",
                "controller": "ShapedDelayedRelativeJointPositionActionCfg",
                "arm_actuator_yaml": FLEXIV_PHYSX_SYSID_ACTUATOR_YAML_PATH,
                "sysid_source": (
                    "output/sysid/flexiv/"
                    "pd_only_gravityoff_high_cmdlimits_seed_default_steps_pm10_physx_n64_g20/best_params.yaml"
                ),
                "physics_freq_hz": FLEXIV_DEPLOYMENT_PHYSICS_FREQ_HZ,
                "decimation": FLEXIV_DEPLOYMENT_DECIMATION,
                "control_freq_hz": FLEXIV_DEPLOYMENT_CONTROL_FREQ_HZ,
                "command_velocity_limit_rad_s": FLEXIV_ROBOT_COLLECTION_COMMAND_VELOCITY_LIMIT,
                "command_acceleration_limit_rad_s2": FLEXIV_ROBOT_COLLECTION_COMMAND_ACCELERATION_LIMIT,
                "sysid_notes": (
                    "PhysX PD-only step-response SysID plus command shaping matched to "
                    "the 50 Hz Flexiv deployment command loop."
                ),
            }
        )
        self.actions.arm_action = ShapedDelayedRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=FLEXIV_ARM_JOINT_NAMES,
            scale=self.joint_action_scale,
            use_zero_offset=True,
            latency_s=FLEXIV_ACTION_LATENCY_MS / 1000.0,
            command_velocity_limit=FLEXIV_ROBOT_COLLECTION_COMMAND_VELOCITY_LIMIT,
            command_acceleration_limit=FLEXIV_ROBOT_COLLECTION_COMMAND_ACCELERATION_LIMIT,
        )

        # Override robot initial pose for ROS inference (fixed pose, no randomization)
        # Joint positions and pos are inherited from parent, only override rotation to be deterministic
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion (x, y, z, w)

        # Override gear base initial pose (fixed pose for ROS inference)
        # Position configured for Rizon 4s workspace
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(0.481, -0.073, -0.005),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Override gear initial poses (fixed poses for ROS inference)
        # Small gear
        self.scene.factory_gear_small.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(0.481, -0.073, -0.005),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Medium gear
        self.scene.factory_gear_medium.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(0.481, -0.073, -0.005),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Large gear
        self.scene.factory_gear_large.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(0.481, -0.073, -0.005),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Fixed asset parameters for ROS inference - derived from configuration
        # These parameters are used by the ROS inference node to validate the environment setup
        # and apply appropriate noise models for robust real-world deployment.
        # Derive position center from gear base init state
        self.fixed_asset_init_pos_center = list(self.scene.factory_gear_base.init_state.pos)
        # Derive position range from parent's randomize_gears_and_base_pose event pose_range
        pose_range = self.events.randomize_gears_and_base_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],  # max value
            pose_range["y"][1],  # max value
            pose_range["z"][1],  # max value
        ]
        # Orientation in degrees (quaternion (0.0, 0.0, 0.70711, -0.70711) = -90° around Z)
        self.fixed_asset_init_orn_deg = [0.0, 0.0, -90.0]
        # Derive orientation range from parent's pose_range (radians to degrees)
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range["roll"][1]),  # convert radians to degrees
            math.degrees(pose_range["pitch"][1]),
            math.degrees(pose_range["yaw"][1]),
        ]
        # Derive observation noise level from parent's gear_shaft_pos noise configuration
        gear_shaft_pos_noise = self.observations.policy.gear_shaft_pos.noise.noise_cfg.n_max
        self.fixed_asset_pos_obs_noise_level = [
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
        ]


@configclass
class Rizon4sGearAssemblyEnvCfg_PLAY(Rizon4sGearAssemblyROSInferenceEnvCfg):
    """Deterministic play/debug configuration for Flexiv Rizon 4s gear assembly.

    Inherits the full ROS-inference configuration and then disables all
    randomization so the simulation is identical on every reset.  Useful for
    comparing simulated and real-world policy behavior at a known pose.

    To debug a specific real-world scenario, edit the constants below to match
    the physical setup, then run::

        python scripts/reinforcement_learning/rsl_rl/play.py \\
            --task IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-Play \\
            --num_envs 1 --checkpoint <path_to_model.pt>

    Observation overrides (``OBS_SHAFT_POS``, ``OBS_SHAFT_QUAT``) let you
    inject fixed values into the policy's observation tensor regardless of
    simulation state.  Set to ``None`` to use the simulated values.
    """

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  SCENE SETUP — edit to match your real-world setup                  ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    GEAR_TYPE: str = "gear_large"
    GEAR_BASE_POS: tuple = (0.481, -0.073, -0.005)
    GEAR_BASE_ROT: tuple = (0.0, 0.0, -0.70711, 0.70711)
    GEAR_Z_OFFSET: float = 0.0675

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  OBSERVATION OVERRIDES — set to None to use simulated values        ║
    # ║                                                                      ║
    # ║  Obs layout: [joint_pos(7) | joint_vel(7) | shaft_pos(3) |          ║
    # ║               shaft_quat(4)]                                         ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    OBS_SHAFT_POS: tuple | None = None  # e.g. (0.481, -0.028, -0.005)
    OBS_SHAFT_QUAT: tuple | None = None  # e.g. (0.0, 0.0, -0.70711, 0.70711)

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # ── Fix gear type (no random selection) ───────────────────────────
        self.events.randomize_gear_type.params["gear_types"] = [self.GEAR_TYPE]

        # ── Override gear base pose ───────────────────────────────────────
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(
            pos=self.GEAR_BASE_POS,
            rot=self.GEAR_BASE_ROT,
        )
        for attr in ("factory_gear_small", "factory_gear_medium", "factory_gear_large"):
            getattr(self.scene, attr).init_state = RigidObjectCfg.InitialStateCfg(
                pos=self.GEAR_BASE_POS,
                rot=self.GEAR_BASE_ROT,
            )

        # ── Zero out all pose randomization ───────────────────────────────
        self.events.randomize_gears_and_base_pose.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
        self.events.randomize_gears_and_base_pose.params["gear_pos_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [self.GEAR_Z_OFFSET, self.GEAR_Z_OFFSET],
        }

        # ── Disable observation noise ─────────────────────────────────────
        self.observations.policy.enable_corruption = False

        # ── Observation overrides (replace terms with constant functions) ─
        if self.OBS_SHAFT_POS is not None:
            self.observations.policy.gear_shaft_pos = ObsTerm(func=constant_obs, params={"value": self.OBS_SHAFT_POS})
        if self.OBS_SHAFT_QUAT is not None:
            self.observations.policy.gear_shaft_quat = ObsTerm(func=constant_obs, params={"value": self.OBS_SHAFT_QUAT})
