# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from isaaclab_physx.physics import PhysxCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import REQUIRED, replace_config

from isaaclab_assets import HUMANOID_28_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@dataclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s: Any = 10.0
    decimation: Any = 2

    # spaces
    observation_space: Any = 81
    action_space: Any = 28
    state_space: Any = 0
    num_amp_observations: Any = 2
    amp_observation_space: Any = 81

    early_termination: Any = True
    termination_height: Any = 0.5

    motion_file: str = REQUIRED
    reference_body: Any = "torso"
    reset_strategy: Any = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            dt=1 / 60,
            render_interval=2,
            physics=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
        )
    )

    # scene
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)
    )

    # robot
    robot: ArticulationCfg = field(
        default_factory=lambda: replace_config(
            replace_config(HUMANOID_28_CFG, prim_path="{ENV_REGEX_NS}/Robot"),
            actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=None,
                    damping=None,
                    joint_velocity_limit={
                        ".*": 100.0,
                    },
                ),
            },
        )
    )


@dataclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    motion_file: Any = field(default_factory=lambda: os.path.join(MOTIONS_DIR, "humanoid_dance.npz"))


@dataclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file: Any = field(default_factory=lambda: os.path.join(MOTIONS_DIR, "humanoid_run.npz"))


@dataclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file: Any = field(default_factory=lambda: os.path.join(MOTIONS_DIR, "humanoid_walk.npz"))
