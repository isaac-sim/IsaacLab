# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .joint_pos_env_cfg import Rizon4sGearAssemblyEnvCfg


def constant_obs(env, value: tuple) -> torch.Tensor:
    """Observation function that returns a fixed tensor every step."""
    return torch.tensor([value], device=env.device, dtype=torch.float32).expand(env.num_envs, -1)


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
            --task Isaac-Deploy-GearAssembly-Rizon4s-Grav-Play-v0 \\
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
