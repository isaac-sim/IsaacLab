# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MDP2RLEnv on a headless Newton Cartpole, built from a local USD file."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv, MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

CARTPOLE_USD = Path(__file__).parents[3] / "isaaclab_newton/test/assets/data/fixed_cartpole.usda"
NUM_ENVS = 16


robot = mdp2.Entity("robot")
cart, pole, joints = robot.joints("slider_to_cart"), robot.joints("cart_to_pole"), robot.joints(".*")


def reset_joints(subset: mdp2.SubsetSelector, position_range: tuple, velocity_range: tuple) -> mdp2.Event:
    """Return an event that offsets the subset's joint state from its default by uniform samples."""
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


def eager_joint_vel_l1(ctx: mdp2.TermContext):
    joint_vel = ctx.inputs["joint_vel"][:, ctx.inputs["joint_ids"].long()]

    # Reading a value on the host fails inside a CUDA graph, which eager terms never run in.
    if joint_vel.isfinite().all().item():
        ctx.out.copy_(joint_vel.abs().sum(dim=-1))


def cartpole_cfg(backend: str = "warp", capture: bool = True, eager: bool = False) -> MDP2RLEnvCfg:
    """The Cartpole task of ``Isaac-Cartpole``, on the fixed Cartpole of the Newton tests, with a 1 s episode.

    With ``eager``, the cart velocity penalty is an eager term.
    """
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

    terminated = {"terminated": mdp2.terminated}
    task = mdp2.TaskCfg(
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
            "alive": mdp2.Reward(mdp.is_alive, inputs=terminated),
            "terminating": mdp2.Reward(mdp.is_terminated, inputs=terminated, weight=-2.0),
            "pole_pos": mdp2.Reward(
                mdp.joint_pos_target_l2,
                inputs={"joint_pos": robot.joint_pos, "joint_ids": pole},
                params={"target": 0.0},
                weight=-1.0,
            ),
            "cart_vel": mdp2.Reward(
                mdp2.Term(eager=eager_joint_vel_l1) if eager else mdp.joint_vel_l1,
                inputs={"joint_vel": robot.joint_vel, "joint_ids": cart},
                weight=-0.01,
            ),
            "pole_vel": mdp2.Reward(
                mdp.joint_vel_l1, inputs={"joint_vel": robot.joint_vel, "joint_ids": pole}, weight=-0.005
            ),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=1.0),
            "cart_out_of_bounds": mdp2.Termination(
                mdp.joint_pos_out_of_manual_limit,
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


def rollout(cfg: MDP2RLEnvCfg, num_steps: int) -> tuple[int, list[dict[str, torch.Tensor]]]:
    """Reset, step random actions, and close; return the graphs that a captured step replays and the step results."""
    env = MDP2RLEnv(cfg)
    try:
        generator = torch.Generator(device=env.device).manual_seed(0)
        # Half the carts are pushed each way, so that they leave their bounds.
        direction = torch.ones(env.num_envs, 1, device=env.device)
        direction[::2] = -1.0

        observations, _ = env.reset()
        history = [{"policy": observations["policy"].clone()}]
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

        num_graphs = env.num_graphs
    finally:
        env.close()
    return num_graphs, history


def shift_root_torch(mask, root_link_pose_w, default_root_pose, env_origins, *, offset):
    pose = default_root_pose.clone()
    pose[:, :3] += env_origins
    pose[:, 0] += offset
    root_link_pose_w.copy_(torch.where(mask[:, None], pose, root_link_pose_w))


def test_env_follows_the_gym_interface_and_commits_resets():
    """Steps follow the Gym interface, and the solver sees the reset joint state and the moved fixed base."""
    cfg = cartpole_cfg()
    cfg.task.events["shift_root"] = mdp2.Event(
        mdp2.Term(torch=shift_root_torch),
        inputs={"default_root_pose": robot.default_root_pose, "env_origins": mdp2.Entity("scene").env_origins},
        outputs={"root_link_pose_w": robot.root_link_pose_w},
        params={"offset": 0.5},
    )
    env = MDP2RLEnv(cfg)
    try:
        robot_data = env.scene["robot"].data
        assert env.single_observation_space["policy"].shape == (4,)
        assert env.single_action_space.shape == (1,)
        assert env.max_episode_length == 60

        observations, extras = env.reset()

        assert observations["policy"].shape == (NUM_ENVS, 4) and extras == {}
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
        # The fixed base stays where the reset moved it, through the solver's steps.
        base_x = env.scene.env_origins[:, 0] + robot_data.default_root_pose.torch[:, 0] + 0.5
        torch.testing.assert_close(robot_data.body_pos_w.torch[:, 0, 0], base_x, atol=1e-5, rtol=0.0)
        assert torch.equal(observations["policy"][:, :2], robot_data.joint_pos.torch)
    finally:
        env.close()


def test_env_rejects_actuators_with_state():
    """Masked resets reset every world's actuators, so actuators that keep state, like delayed ones, are refused."""
    cfg = cartpole_cfg()
    cfg.scene.robot.actuators["cart"] = DelayedPDActuatorCfg(
        joint_names_expr=["slider_to_cart"], joint_effort_limit=400.0, stiffness=0.0, damping=10.0, max_delay=2
    )

    with pytest.raises(ValueError, match=r"'robot' has the actuator groups \['cart'\]"):
        MDP2RLEnv(cfg)


def test_captured_steps_match_eager_steps_and_the_torch_backend():
    """A captured step replays one graph, or graphs around the eager terms, and gives bitwise the eager results."""
    num_steps = 80  # Past the time-out of every world.
    num_graphs, captured_run = rollout(cartpole_cfg(), num_steps)
    eager_graphs, eager_run = rollout(cartpole_cfg(capture=False), num_steps)
    assert num_graphs == 1 and eager_graphs == 0
    assert any(step["terminated"].any() for step in captured_run[1:])

    for backend in ("warp", "torch"):
        backend_graphs, backend_run = rollout(cartpole_cfg(backend=backend, eager=True), num_steps)
        _, backend_eager_run = rollout(cartpole_cfg(backend=backend, capture=False, eager=True), num_steps)
        # The eager reward splits evaluate: physics and the terms before it, then the rest of the step.
        assert backend_graphs == 2
        for captured_step, eager_step, backend_step in zip(captured_run, eager_run, backend_run):
            for key, value in captured_step.items():
                assert torch.equal(value, eager_step[key])
                torch.testing.assert_close(backend_step[key], value, atol=1e-4, rtol=1e-4)
        for captured_step, eager_step in zip(backend_run, backend_eager_run):
            for key, value in captured_step.items():
                assert torch.equal(value, eager_step[key])


def test_env_splits_the_worlds_between_tasks_and_logs_terms_where_they_run():
    cfg = cartpole_cfg()
    without_alive = replace(cfg.task, rewards={**cfg.task.rewards, "alive": None})
    cfg.task = [cfg.task, without_alive]
    env = MDP2RLEnv(cfg)
    try:
        assert [config.worlds for config in env.mdp.world_configs] == [range(0, 8), range(8, 16)]

        env.reset()
        for _ in range(env.max_episode_length):
            _, _, _, _, extras = env.step(torch.zeros(NUM_ENVS, 1, device=env.device))

        # Unpushed carts stay in bounds, so every world was alive for its whole episode.
        assert extras["log"]["Episode_Termination/cart_out_of_bounds"] == 0.0
        assert extras["log"]["Episode_Reward/alive"].item() == pytest.approx(1.0)
    finally:
        env.close()
