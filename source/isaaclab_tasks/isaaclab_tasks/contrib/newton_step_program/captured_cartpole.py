# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A cart-pole MDP whose whole environment step, physics included, is one CUDA graph.

The MDP mirrors ``Isaac-Cartpole`` (effort action, reward terms and weights, terminations, reset ranges, relative
joint observations) with Warp kernels, mask-based partial resets, and the Newton step program recorded into the same
capture. It demonstrates the contract a graph-captured MDP needs from the physics layer: prepare the step program,
then record ``sim.step()`` inside the caller's capture.
"""

from __future__ import annotations

import math

import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel(enable_backward=False)
def _apply_effort(
    actions: wp.array2d(dtype=wp.float32), cart: int, scale: float, joint_f: wp.array2d(dtype=wp.float32)
):
    i = wp.tid()
    joint_f[i, cart] = scale * actions[i, 0]


@wp.kernel(enable_backward=False)
def _evaluate(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    default_pos: wp.array2d(dtype=wp.float32),
    default_vel: wp.array2d(dtype=wp.float32),
    cart: int,
    pole: int,
    max_episode_length: int,
    step_dt: float,
    seed: wp.array(dtype=wp.int32),
    episode_length: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    reset_mask: wp.array(dtype=wp.bool),
    reset_pos: wp.array2d(dtype=wp.float32),
    reset_vel: wp.array2d(dtype=wp.float32),
):
    """Advance episode counters, score the step, and draw reset states for ending worlds."""
    i = wp.tid()
    length = episode_length[i] + 1
    out_of_bounds = wp.abs(joint_pos[i, cart]) > 3.0
    timed_out = length >= max_episode_length
    pole_pos = joint_pos[i, pole]
    value = 1.0 - 2.0 * float(out_of_bounds)
    value -= pole_pos * pole_pos
    value -= 0.01 * wp.abs(joint_vel[i, cart]) + 0.005 * wp.abs(joint_vel[i, pole])
    reward[i] = value * step_dt
    terminated[i] = out_of_bounds
    truncated[i] = timed_out
    done = out_of_bounds or timed_out
    reset_mask[i] = done
    episode_length[i] = wp.where(done, 0, length)
    # Reset offsets follow Isaac-Cartpole's reset_joints_by_offset ranges.
    rng = wp.rand_init(seed[0], i)
    reset_pos[i, cart] = default_pos[i, cart] + wp.randf(rng, -1.0, 1.0)
    reset_vel[i, cart] = default_vel[i, cart] + wp.randf(rng, -0.5, 0.5)
    reset_pos[i, pole] = default_pos[i, pole] + wp.randf(rng, -0.25 * wp.pi, 0.25 * wp.pi)
    reset_vel[i, pole] = default_vel[i, pole] + wp.randf(rng, -0.25 * wp.pi, 0.25 * wp.pi)


@wp.kernel(enable_backward=False)
def _observe(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    default_pos: wp.array2d(dtype=wp.float32),
    default_vel: wp.array2d(dtype=wp.float32),
    num_joints: int,
    seed: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=wp.float32),
):
    """Write relative joint positions and velocities, then advance the random stream once per step."""
    i = wp.tid()
    for j in range(num_joints):
        obs[i, j] = joint_pos[i, j] - default_pos[i, j]
        obs[i, num_joints + j] = joint_vel[i, j] - default_vel[i, j]
    if i == 0:
        seed[0] = seed[0] + 1


class CapturedCartpole:
    """Run ``Isaac-Cartpole``'s MDP as Warp stages around the Newton step program, capturable as one graph.

    The environment provides the scene and simulation; this class replaces its managers. Write actions into
    :attr:`actions`, then call :meth:`step` (eager) or :meth:`replay` (captured). Rewards, termination and truncation
    flags, and next observations land in fixed buffers.
    """

    def __init__(self, env: ManagerBasedRLEnv, seed: int = 0):
        """Bind the MDP to an environment's robot.

        Args:
            env: ``Isaac-Cartpole`` environment on Newton physics.
            seed: Seed of the reset random stream.
        """
        self.env = env
        self.robot = env.scene["robot"]
        if self.robot.data.has_joint_ordering:
            raise ValueError("CapturedCartpole writes backend joint buffers and requires identity joint ordering.")
        device, num_envs = env.device, env.num_envs
        self.num_joints = self.robot.num_joints
        self.cart = self.robot.find_joints("slider_to_cart")[0][0]
        self.pole = self.robot.find_joints("cart_to_pole")[0][0]
        self.max_episode_length = math.ceil(env.cfg.episode_length_s / env.step_dt)
        self.step_dt = env.step_dt
        self.actions = wp.zeros((num_envs, 1), dtype=wp.float32, device=device)
        self.obs = wp.zeros((num_envs, 2 * self.num_joints), dtype=wp.float32, device=device)
        self.reward = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self.terminated = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.truncated = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self.episode_length = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self._reset_mask = wp.zeros(num_envs, dtype=wp.bool, device=device)
        self._reset_pos = wp.zeros((num_envs, self.num_joints), dtype=wp.float32, device=device)
        self._reset_vel = wp.zeros((num_envs, self.num_joints), dtype=wp.float32, device=device)
        self._seed = wp.array([seed], dtype=wp.int32, device=device)
        self.graph: wp.Graph | None = None

    def step(self) -> None:
        """Advance one environment step: apply actions, step physics, score, reset ending worlds, and observe."""
        data, device, num_envs = self.robot.data, self.env.device, self.env.num_envs
        default_pos, default_vel = data.default_joint_pos.warp, data.default_joint_vel.warp
        wp.launch(
            _apply_effort, num_envs, [self.actions, self.cart, 100.0], [data._sim_bind_joint_effort], device=device
        )
        self.env.sim.step(render=False)
        wp.launch(
            _evaluate,
            num_envs,
            [
                data.joint_pos.warp,
                data.joint_vel.warp,
                default_pos,
                default_vel,
                self.cart,
                self.pole,
                self.max_episode_length,
                self.step_dt,
                self._seed,
            ],
            [
                self.episode_length,
                self.reward,
                self.terminated,
                self.truncated,
                self._reset_mask,
                self._reset_pos,
                self._reset_vel,
            ],
            device=device,
        )
        self.robot.write_joint_state_to_sim_mask(
            position=self._reset_pos, velocity=self._reset_vel, env_mask=self._reset_mask
        )
        wp.launch(
            _observe,
            num_envs,
            [data.joint_pos.warp, data.joint_vel.warp, default_pos, default_vel, self.num_joints, self._seed],
            [self.obs],
            device=device,
        )

    def capture(self) -> wp.Graph:
        """Record :meth:`step` into one CUDA graph, including the Newton step program.

        Prepares the step program first, so recording allocates nothing and never nests a capture.
        """
        NewtonManager.prepare()
        with wp.ScopedCapture(device=self.env.device) as capture:
            self.step()
        self.graph = capture.graph
        return self.graph

    def replay(self) -> None:
        """Advance one environment step by replaying the captured graph."""
        wp.capture_launch(self.graph)
