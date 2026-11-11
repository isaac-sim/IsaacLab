# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MDP2RLEnv on a headless Newton Cartpole, built from a local USD file."""

import math
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv, MDP2RLEnvCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

CARTPOLE_USD = Path(__file__).parents[3] / "isaaclab_newton/test/assets/data/fixed_cartpole.usda"
NUM_ENVS = 16


@wp.func
def joint_pos_rel_warp(
    world: int, joint: int, joint_pos: wp.array2d(dtype=float), default_joint_pos: wp.array2d(dtype=float)
) -> float:
    return joint_pos[world, joint] - default_joint_pos[world, joint]


@wp.func
def is_alive_warp(world: int, joint_pos: wp.array2d(dtype=float)) -> float:
    return 1.0


@wp.func
def joint_pos_target_l2_warp(
    world: int, joint_pos: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32), target: float
) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        error = joint_pos[world, joint_ids[k]] - target
        total += error * error
    return total


@wp.func
def joint_vel_l1_warp(world: int, joint_vel: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)) -> float:
    total = float(0.0)
    for k in range(joint_ids.shape[0]):
        total += wp.abs(joint_vel[world, joint_ids[k]])
    return total


@wp.func
def joint_pos_out_of_manual_limit_warp(
    world: int, joint_pos: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32), bounds: wp.vec2
) -> bool:
    outside = bool(False)
    for k in range(joint_ids.shape[0]):
        value = joint_pos[world, joint_ids[k]]
        outside = outside or value < bounds[0] or value > bounds[1]
    return outside


@wp.func
def reset_joints_by_offset_warp(
    world: int,
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    default_joint_pos: wp.array2d(dtype=float),
    default_joint_vel: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    position_range: wp.vec2,
    velocity_range: wp.vec2,
    rng: wp.int32,
):
    count = joint_ids.shape[0]
    for k in range(count):
        joint = joint_ids[k]
        position = wp.randf(wp.rand_init(rng, k), position_range[0], position_range[1])
        velocity = wp.randf(wp.rand_init(rng, count + k), velocity_range[0], velocity_range[1])
        joint_pos[world, joint] = default_joint_pos[world, joint] + position
        joint_vel[world, joint] = default_joint_vel[world, joint] + velocity


def reset_joints_by_offset_torch(
    mask, joint_pos, joint_vel, default_joint_pos, default_joint_vel, joint_ids, rng, *, position_range, velocity_range
):
    ids = joint_ids.long()
    position = mdp2.uniform(rng, *position_range, ids.shape)
    velocity = mdp2.uniform(rng, *velocity_range, ids.shape, offset=len(ids))
    joint_pos[:, ids] = torch.where(mask[:, None], default_joint_pos[:, ids] + position, joint_pos[:, ids])
    joint_vel[:, ids] = torch.where(mask[:, None], default_joint_vel[:, ids] + velocity, joint_vel[:, ids])


joint_pos_rel = mdp2.Term(
    torch=lambda joint_pos, default_joint_pos: joint_pos - default_joint_pos,
    warp=joint_pos_rel_warp,
    shape=("worlds", "joints"),
)
is_alive = mdp2.Term(torch=lambda joint_pos: torch.ones_like(joint_pos[:, 0]), warp=is_alive_warp)
joint_pos_target_l2 = mdp2.Term(
    torch=lambda joint_pos, joint_ids, *, target: (joint_pos[:, joint_ids.long()] - target).square().sum(-1),
    warp=joint_pos_target_l2_warp,
)
joint_vel_l1 = mdp2.Term(
    torch=lambda joint_vel, joint_ids: joint_vel[:, joint_ids.long()].abs().sum(-1), warp=joint_vel_l1_warp
)
joint_pos_out_of_manual_limit = mdp2.Term(
    torch=lambda joint_pos, joint_ids, *, bounds: (
        (joint_pos[:, joint_ids.long()] < bounds[0]) | (joint_pos[:, joint_ids.long()] > bounds[1])
    ).any(-1),
    warp=joint_pos_out_of_manual_limit_warp,
)
reset_joints_by_offset = mdp2.Term(torch=reset_joints_by_offset_torch, warp=reset_joints_by_offset_warp, random=True)

robot = mdp2.Entity("robot")
cart, pole = robot.joints("slider_to_cart"), robot.joints("cart_to_pole")


def reset_joints(joints: mdp2.SubsetSelector, position_range: tuple, velocity_range: tuple) -> mdp2.Event:
    return mdp2.Event(
        reset_joints_by_offset,
        inputs={
            "default_joint_pos": robot.default_joint_pos,
            "default_joint_vel": robot.default_joint_vel,
            "joint_ids": joints,
        },
        outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel},
        params={"position_range": position_range, "velocity_range": velocity_range},
    )


def cartpole_cfg(backend: str = "warp", capture: bool = True) -> MDP2RLEnvCfg:
    """The Cartpole task of ``Isaac-Cartpole``, on the fixed Cartpole of the Newton tests."""
    scene = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=4.0)
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(CARTPOLE_USD)),
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=["slider_to_cart"], joint_effort_limit=400.0, stiffness=0.0, damping=10.0
            ),
            "pole": ImplicitActuatorCfg(
                joint_names_expr=["cart_to_pole"], joint_effort_limit=400.0, stiffness=0.0, damping=0.0
            ),
        },
    )
    joint_state = {"default_joint_pos": robot.default_joint_pos}
    task = mdp2.TaskCfg(
        actions={"joint_effort": mdp2.DirectAction(robot.joint_effort_target, subset=cart, scale=100.0)},
        observations={
            "policy": mdp2.ObservationGroup(
                joint_pos_rel=mdp2.Observation(joint_pos_rel, inputs={"joint_pos": robot.joint_pos, **joint_state}),
                joint_vel_rel=mdp2.Observation(
                    joint_pos_rel,
                    inputs={"joint_pos": robot.joint_vel, "default_joint_pos": robot.default_joint_vel},
                ),
            )
        },
        rewards={
            "alive": mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}),
            "pole_pos": mdp2.Reward(
                joint_pos_target_l2,
                inputs={"joint_pos": robot.joint_pos, "joint_ids": pole},
                params={"target": 0.0},
                weight=-1.0,
            ),
            "cart_vel": mdp2.Reward(
                joint_vel_l1, inputs={"joint_vel": robot.joint_vel, "joint_ids": cart}, weight=-0.01
            ),
            "pole_vel": mdp2.Reward(
                joint_vel_l1, inputs={"joint_vel": robot.joint_vel, "joint_ids": pole}, weight=-0.005
            ),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=1.0),
            "cart_out_of_bounds": mdp2.Termination(
                joint_pos_out_of_manual_limit,
                inputs={"joint_pos": robot.joint_pos, "joint_ids": cart},
                params={"bounds": (-1.5, 1.5)},
            ),
        },
        events={
            "reset_cart_position": reset_joints(cart, (-1.0, 1.0), (-0.5, 0.5)),
            "reset_pole_position": reset_joints(
                pole, (-0.25 * math.pi, 0.25 * math.pi), (-0.25 * math.pi, 0.25 * math.pi)
            ),
        },
    )
    return MDP2RLEnvCfg(
        sim=SimulationCfg(dt=1 / 120, device="cuda:0", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        scene=scene,
        task=task,
        decimation=2,
        backend=mdp2.Backend(backend),
        seed=42,
        capture=capture,
    )


def rollout(cfg: MDP2RLEnvCfg, num_steps: int) -> tuple[MDP2RLEnv | None, list[dict[str, torch.Tensor]]]:
    """Reset, then step random actions, recording the step results. Closes the environment."""
    env = MDP2RLEnv(cfg)
    try:
        generator = torch.Generator(device=env.device).manual_seed(0)
        history = []
        observations, _ = env.reset()
        history.append({"policy": observations["policy"].clone()})
        # Half the carts are pushed each way, so that they leave their bounds.
        direction = torch.ones(env.num_envs, 1, device=env.device)
        direction[::2] = -1.0
        for _ in range(num_steps):
            actions = direction * (torch.rand(env.num_envs, 1, device=env.device, generator=generator) + 0.5)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            history.append(
                {
                    "policy": observations["policy"].clone(),
                    "rewards": rewards.clone(),
                    "terminated": terminated.clone(),
                    "truncated": truncated.clone(),
                }
            )
        is_captured = env.is_captured
    finally:
        env.close()
    return is_captured, history


def test_env_follows_the_gym_interface_and_commits_resets():
    env = MDP2RLEnv(cartpole_cfg())
    try:
        assert env.single_observation_space["policy"].shape == (4,)
        assert env.single_action_space.shape == (1,)
        assert env.max_episode_length == 60

        observations, extras = env.reset()
        assert observations["policy"].shape == (NUM_ENVS, 4) and extras == {}
        robot_data = env.scene["robot"].data
        assert (robot_data.joint_pos.torch[:, 0].abs() <= 1.0).all() and robot_data.joint_pos.torch.std(0).min() > 0

        ended = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        for _ in range(env.max_episode_length):
            observations, rewards, terminated, truncated, _ = env.step(torch.ones(NUM_ENVS, 1, device=env.device))
            assert rewards.dtype == torch.float32 and terminated.dtype == truncated.dtype == torch.bool
            assert (env.scene["robot"].actuators.target_command.effort.torch[:, 0] == 100.0).all()
            ended |= terminated | truncated
            # Reset worlds start from their new joint state, which the solver's body poses follow.
            cart_x = robot_data.body_pos_w.torch[:, 1, 0] - robot_data.body_pos_w.torch[:, 0, 0]
            torch.testing.assert_close(cart_x, robot_data.joint_pos.torch[:, 0], atol=1e-5, rtol=0.0)
        assert env.is_captured and ended.all()
        assert torch.equal(observations["policy"][:, :2], robot_data.joint_pos.torch)
    finally:
        env.close()


def test_captured_steps_match_eager_steps_and_the_torch_backend():
    num_steps = 80  # past the time-out of every world
    captured, captured_run = rollout(cartpole_cfg(), num_steps)
    eager, eager_run = rollout(cartpole_cfg(capture=False), num_steps)
    _, torch_run = rollout(cartpole_cfg(backend="torch"), num_steps)

    assert captured and not eager
    assert any(step["terminated"].any() for step in captured_run[1:])
    for captured_step, eager_step, torch_step in zip(captured_run, eager_run, torch_run):
        for key, value in captured_step.items():
            assert torch.equal(value, eager_step[key])
            torch.testing.assert_close(torch_step[key], value, atol=1e-4, rtol=1e-4)
