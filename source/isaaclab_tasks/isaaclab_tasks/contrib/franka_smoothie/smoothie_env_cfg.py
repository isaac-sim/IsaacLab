# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-only smoothie task at a fixed 50 Hz control rate."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.configclass import configclass

from ..franka_pour.pour_env_cfg import ActionsCfg
from .scene_cfg import TapSceneCfg
from .smoothie_asset import BUTTON_TOP_POSITION_M, CUP_STATION_POSITION_M
from .smoothie_env import EventsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg
from .smoothie_gripper import SmoothieGripperPositionActionCfg


@configclass
class SmoothieActionsCfg(ActionsCfg):
    """Filtered robot commands with six-phase finger targets [m]."""

    gripper_action = SmoothieGripperPositionActionCfg(asset_name="robot", joint_names=["panda_finger.*"])


@configclass
class FrankaSmoothieEnvCfg(ManagerBasedRLEnvCfg):
    """Ordered fruit, tap, threaded lid, docking and final button task.

    The liquid fill is a task scalar and a renderer mesh, with no liquid mass or
    dynamics.
    """

    scene = TapSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    actions = SmoothieActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    events = EventsCfg()
    terminations = TerminationsCfg()
    basket_grasp_distance: float = 0.04
    """Maximum TCP distance from the basket grasp point that still counts as grasped [m]."""
    basket_relative_speed: float = 0.15
    """Maximum basket-to-hand translational speed for a stable grasp [m/s]."""
    basket_grasp_offset: tuple[float, float, float] = (0.059 / 2**0.5, 0.059 / 2**0.5, 0.090)
    """Basket-local grasp position [m]."""
    basket_grasp_rotation: tuple[float, float, float, float] = (
        math.cos(3 * math.pi / 8),
        math.sin(3 * math.pi / 8),
        0.0,
        0.0,
    )
    """Basket-relative hand quaternion in XYZW order."""
    tap_cup_position: tuple[float, float, float] = CUP_STATION_POSITION_M
    """Cup support position relative to the environment origin [m]."""
    tap_button_position: tuple[float, float, float] = BUTTON_TOP_POSITION_M
    """Unpressed tap button face position [m]."""
    tap_button_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    tap_button_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        self.episode_length_s = 600.0
        self.sim.use_newton_actuators = True
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                use_mujoco_contacts=False,
                integrator="implicitfast",
                nconmax=200,
                njmax=300,
                cone="elliptic",
                impratio=100.0,
                tolerance=1e-6,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=8192),
        )

        dt = 0.01
        decimation = 2
        substeps = 2
        iterations = 20
        line_search = 10
        self.sim.dt, self.decimation = dt, decimation
        self.sim.render_interval = decimation
        self.sim.physics.num_substeps = substeps
        self.sim.physics.solver_cfg.iterations = iterations
        self.sim.physics.solver_cfg.ls_iterations = line_search
