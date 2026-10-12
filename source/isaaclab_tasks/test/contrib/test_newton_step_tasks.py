# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior of tasks that run controllers inside the Newton step."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

# Every task here runs on Newton, which needs no Kit runtime.
launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import gymnasium as gym
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_OSC_TASK = "IsaacContrib-NewtonStep-Reach-Franka-NewtonOSC"


def _make_env(task: str, num_envs: int, seed: int, configure=None) -> gym.Env:
    """Create a Newton MJWarp environment of a registered task on a new stage; ``configure`` edits its cfg."""
    sim_utils.create_new_stage()
    env_cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if configure is not None:
        configure(env_cfg)
    env = gym.make(task, cfg=env_cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None
    return env


def _rollout_osc(num_steps: int, native: bool = True) -> tuple[torch.Tensor, bool]:
    """Step the Newton OSC reach task with fixed pose targets and return the arm trajectory and loop ownership."""

    def configure(env_cfg):
        env_cfg.sim.use_newton_actuators = native
        if not native:
            joints = [f"panda_joint{i}" for i in range(7, 0, -1)] + ["panda_finger_joint1", "panda_finger_joint2"]
            env_cfg.scene.robot.joint_ordering = joints

    env = _make_env(_OSC_TASK, num_envs=4, seed=7, configure=configure)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        generator = torch.Generator(device="cuda:0").manual_seed(0)
        # Absolute root-frame pose targets around a reachable pose with the hand pointing down.
        center = torch.tensor([0.45, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0], device="cuda:0")
        trajectory = []
        with torch.inference_mode():
            for step in range(num_steps):
                # Start without wrenches, then apply both after graph capture, which must rebuild the step graph.
                force = torch.zeros((env.unwrapped.num_envs, robot.num_bodies, 3), device="cuda:0")
                force[:, -1, 0] = 2.0 if step % 6 >= 3 else 0.0
                if step >= 3:
                    robot.permanent_wrench_composer.set_forces_and_torques_index(forces=force, is_global=True)
                if step % 5 == 3:
                    robot.instantaneous_wrench_composer.set_forces_and_torques_index(forces=-force, is_global=False)
                offset = 0.1 * (torch.rand(env.unwrapped.num_envs, 3, device="cuda:0", generator=generator) - 0.5)
                actions = center.repeat(env.unwrapped.num_envs, 1)
                actions[:, :3] += offset
                env.step(actions)
                trajectory.append(robot.data.joint_pos.torch.clone())
        if env.unwrapped._physics_handles_actions:
            # The environment records the physics step, controller included, into its own step graph.
            graph = env.unwrapped.step_graph
            assert graph.captured
            assert all(op.graphable for op in graph.ops if op.name.startswith("action"))
        return torch.stack(trajectory), env.unwrapped._physics_handles_decimation
    finally:
        env.close()


@pytest.mark.parametrize("native", [True, False], ids=["newton-actuators", "torch-actuators-reordered"])
def test_newton_osc_in_step_matches_host_controller(monkeypatch: pytest.MonkeyPatch, native):
    """The captured, physics-step controller tracks targets exactly like the same controller run on the host.

    On the host the term must run before every physics step, so the environment drives the decimation loop; as a
    control callback, physics runs the whole loop in one call.
    """
    in_step, physics_loop = _rollout_osc(num_steps=20, native=native)
    with monkeypatch.context() as patch:
        patch.setattr(NewtonManager, "bind_control", lambda *args: False)
        on_host, host_physics_loop = _rollout_osc(num_steps=20, native=native)

    assert physics_loop and not host_physics_loop
    # The arm moves toward the targets, so the comparison is not between two resting trajectories.
    assert (on_host[-1] - on_host[0]).abs().max() > 0.05
    torch.testing.assert_close(in_step, on_host, atol=1e-4, rtol=0.0)


def _rollout_joint_actions(num_steps: int) -> tuple[torch.Tensor, bool]:
    """Step the joint-position reach task with new random actions every step; return the arm trajectory."""
    env = _make_env("Isaac-Reach-Franka", num_envs=4, seed=7)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        generator = torch.Generator(device="cuda:0").manual_seed(0)
        trajectory = []
        with torch.inference_mode():
            for _ in range(num_steps):
                actions = torch.rand(env.action_space.shape, device="cuda:0", generator=generator) - 0.5
                env.step(actions)
                trajectory.append(robot.data.joint_pos.torch.clone())
        return torch.stack(trajectory), env.unwrapped._physics_handles_actions
    finally:
        env.close()


def test_bound_joint_actions_follow_each_new_action(monkeypatch: pytest.MonkeyPatch):
    """Joint actions applied inside the captured step track every new action, exactly like the host path."""
    in_step, bound = _rollout_joint_actions(num_steps=12)
    with monkeypatch.context() as patch:
        patch.setattr(NewtonManager, "bind_control", lambda *args: False)
        on_host, host_bound = _rollout_joint_actions(num_steps=12)

    assert bound and not host_bound
    assert (on_host[-1] - on_host[0]).abs().max() > 0.05
    torch.testing.assert_close(in_step, on_host, atol=1e-4, rtol=0.0)


def _rollout_env(task: str, captured: bool, num_steps: int) -> tuple[list[torch.Tensor], object]:
    """Step a manager-based task deterministically with random actions and short episodes."""

    def configure(env_cfg):
        env_cfg.capture_step = captured
        env_cfg.sim.physics.deterministic_mode = "run_to_run"
        env_cfg.observations.policy.enable_corruption = False
        # Short episodes guarantee resets inside the rollout.
        env_cfg.episode_length_s = 8 * env_cfg.decimation * env_cfg.sim.dt

    env = _make_env(task, num_envs=16, seed=5, configure=configure)
    try:
        env.reset()
        generator = torch.Generator(device="cuda:0").manual_seed(0)
        shape = env.action_space.shape
        state = env.unwrapped.sim.physics_manager.backend.state_0
        robot = env.unwrapped.scene["robot"]
        trajectory = []
        with torch.inference_mode():
            for _ in range(num_steps):
                obs, reward, terminated, truncated, extras = env.step(
                    torch.rand(shape, device="cuda:0", generator=generator)
                )
                joints = torch.cat([wp.to_torch(state.joint_q), wp.to_torch(state.joint_qd)])
                # Lazily computed asset data read after the step, outside any graph, must be current too.
                bodies = robot.data.body_com_pose_w.torch.flatten()
                # Logged metrics come from buffers that terms replace every step.
                logs = [torch.as_tensor(extras["log"][key], device="cuda:0").flatten() for key in sorted(extras["log"])]
                step = (obs["policy"].flatten(), reward, terminated, truncated, joints, bodies, *logs)
                trajectory.append(torch.cat(step).clone())
        return trajectory, env.unwrapped.step_graph
    finally:
        env.close()


@pytest.mark.parametrize("task", ["Isaac-Cartpole", "Isaac-Reach-Franka", "Isaac-Velocity-Flat-AnymalD"])
def test_captured_environment_step_matches_eager(task):
    """Replaying the environment step from CUDA graphs reproduces eager stepping exactly, resets included.

    Reach-Franka keeps its host-scheduled curriculum and manager reset as eager islands, and ANYmal-D its TorchScript
    LSTM actuators and their state reset, between the captured segments.
    """
    eager, eager_graph = _rollout_env(task, captured=False, num_steps=24)
    captured, step_graph = _rollout_env(task, captured=True, num_steps=24)

    assert eager_graph is None
    assert step_graph is not None and step_graph.captured
    for eager_step, captured_step in zip(eager, captured):
        # Reductions may round differently once recorded; anything beyond rounding is a missed update.
        torch.testing.assert_close(captured_step, eager_step, atol=1e-5, rtol=0.0)
