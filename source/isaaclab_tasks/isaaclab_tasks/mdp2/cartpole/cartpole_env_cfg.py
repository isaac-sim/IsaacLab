# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole balancing: push a cart along a rail to keep the pole on it upright."""

import math

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

robot = mdp2.Entity("robot")
cart = robot.joints("slider_to_cart")
pole = robot.joints("cart_to_pole")
joints = robot.joints(".*")


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """A cart-pole on a ground plane, lit by a distant light."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = replace(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        # Roll, pitch, and yaw of (0, -45, -45) degrees, as (x, y, z, w).
        init_state=AssetBaseCfg.InitialStateCfg(rot=(-0.1464466, -0.3535534, -0.3535534, 0.8535534)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
    )


@configclass
class CartpolePhysicsCfg(PresetCfg):
    """Newton physics presets of the Cartpole task."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3, cone="pyramidal", impratio=1, integrator="implicitfast"),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True), debug_mode=False, use_cuda_graph=True
    )
    default: NewtonCfg = newton_mjwarp


def reset_joints(subset: mdp2.SubsetSelector, position_range: tuple, velocity_range: tuple) -> mdp2.Event:
    """Offset the default state of some joints by uniform samples."""
    return mdp2.Event(
        mdp.reset_joints_by_offset,
        inputs={
            "default_joint_pos": robot.default_joint_pos,
            "default_joint_vel": robot.default_joint_vel,
            "soft_joint_pos_limits": robot.soft_joint_pos_limits,
            "soft_joint_vel_limits": robot.soft_joint_vel_limits,
            "joint_ids": subset,
        },
        outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel},
        params={"position_range": position_range, "velocity_range": velocity_range},
    )


TASK = mdp2.TaskCfg(
    actions={"joint_effort": mdp2.DirectAction(robot.joint_effort_target, subset=cart, scale=100.0)},
    observations={
        "policy": mdp2.ObservationGroup(
            joint_pos_rel=mdp2.Observation(
                mdp.joint_pos_rel,
                inputs={
                    "joint_pos": robot.joint_pos,
                    "default_joint_pos": robot.default_joint_pos,
                    "joint_ids": joints,
                },
            ),
            joint_vel_rel=mdp2.Observation(
                mdp.joint_vel_rel,
                inputs={
                    "joint_vel": robot.joint_vel,
                    "default_joint_vel": robot.default_joint_vel,
                    "joint_ids": joints,
                },
            ),
        )
    },
    rewards={
        "alive": mdp2.Reward(mdp.is_alive, inputs={"terminated": mdp2.terminated}, weight=1.0),
        "terminating": mdp2.Reward(mdp.is_terminated, inputs={"terminated": mdp2.terminated}, weight=-2.0),
        "pole_pos": mdp2.Reward(
            mdp.joint_pos_target_l2,
            inputs={"joint_pos": robot.joint_pos, "joint_ids": pole},
            params={"target": 0.0},
            weight=-1.0,
        ),
        "cart_vel": mdp2.Reward(
            mdp.joint_vel_l1, inputs={"joint_vel": robot.joint_vel, "joint_ids": cart}, weight=-0.01
        ),
        "pole_vel": mdp2.Reward(
            mdp.joint_vel_l1, inputs={"joint_vel": robot.joint_vel, "joint_ids": pole}, weight=-0.005
        ),
    },
    terminations={
        "time_out": mdp2.TimeLimit(episode_length_s=5.0),
        "cart_out_of_bounds": mdp2.Termination(
            mdp.joint_pos_out_of_manual_limit,
            inputs={"joint_pos": robot.joint_pos, "joint_ids": cart},
            params={"bounds": (-3.0, 3.0)},
        ),
    },
    events={
        "reset_cart_position": reset_joints(cart, (-1.0, 1.0), (-0.5, 0.5)),
        "reset_pole_position": reset_joints(pole, (-0.25 * math.pi, 0.25 * math.pi), (-0.25 * math.pi, 0.25 * math.pi)),
    },
)
"""The MDP of the task."""


@configclass
class CartpoleEnvCfg(MDP2RLEnvCfg):
    """Configuration of the Cartpole environment."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1 / 120, render_interval=2, physics=CartpolePhysicsCfg())
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    task: mdp2.TaskCfg = TASK
    decimation: int = 2
