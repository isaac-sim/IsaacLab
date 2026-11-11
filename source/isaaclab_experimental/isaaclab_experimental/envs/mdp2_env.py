# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environments that run an MDP2 task on an interactive scene, in place of the managers of manager-based ones."""

from __future__ import annotations

import functools
import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics.newton_backend import capture_graph

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import use_stage
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.warp import ProxyArray

from isaaclab_experimental import mdp2

from .mdp2_env_cfg import MDP2EnvCfg, MDP2RLEnvCfg
from .mdp2_scene_data import InteractiveSceneData


class MDP2Env:
    """An interactive scene driven by an MDP2 task, like a :class:`~isaaclab.envs.ManagerBasedEnv` without managers.

    A control step applies the actions, steps the physics ``decimation`` times, evaluates the MDP, resets the
    worlds whose episodes ended, and observes them. Resets are masks over the worlds, so with the Warp backend and
    a fully capturable MDP, every step after the first replays as one CUDA graph, physics included. Newton only.
    """

    diagnostics: ClassVar[bool] = False  # Whether evaluate writes the reward and termination of every term

    def __init__(self, cfg: MDP2EnvCfg):
        self.cfg = cfg
        if cfg.seed is not None:
            configure_seed(cfg.seed)
        if SimulationContext.instance() is not None:
            raise RuntimeError("A simulation context already exists. Close the environment that created it first.")

        self.sim = SimulationContext(cfg.sim)
        try:
            self._init_sim()
        except Exception:
            self.sim.clear_instance()
            raise
        self._is_closed = False

    def _init_sim(self) -> None:
        """Build the scene and the MDP; on failure, the caller clears the simulation context."""
        cfg, sim = self.cfg, self.sim

        # Terms read sensor data in kernels, so sensors refresh every step rather than when their data is read.
        cfg.scene.lazy_sensor_update = False
        with use_stage(sim.stage):
            self.scene = InteractiveScene(cfg.scene)
        sim.register_interactive_scene(self.scene)
        with use_stage(sim.stage):
            sim.reset()
        self.scene.update(dt=self.physics_dt)

        sim.backend_manager.set_decimation(cfg.decimation)
        # Newton-native actuators step all substeps in one call.
        self._substeps = cfg.decimation if sim.backend_manager.handles_decimation() else 1

        self.scene_data = InteractiveSceneData(self.scene, self.step_dt)
        schema = self.scene_data.schema
        builder = mdp2.MDPBuilder(schema)
        if isinstance(cfg.task, mdp2.TaskCfg):
            builder.add_task(cfg.task)
        else:
            # The first tasks get one more world when the worlds do not split evenly.
            for index, task in enumerate(cfg.task):
                task_builder = mdp2.MDPBuilder(schema)
                task_builder.add_task(task)
                num_worlds = schema.num_worlds // len(cfg.task) + (index < schema.num_worlds % len(cfg.task))
                builder.replicate(task_builder, num_worlds)
        self.mdp = builder.finalize(device=self.device, backend=cfg.backend, diagnostics=self.diagnostics)
        self.state = self.mdp.state(self.scene_data, seed=cfg.seed or 0)
        mdp2.startup(self.mdp, self.state)
        self.transition = self.mdp.transition()

        self.actions = ProxyArray(wp.zeros((self.num_envs, self.mdp.action_dim), dtype=wp.float32, device=self.device))
        self._all_envs = ProxyArray(wp.ones(self.num_envs, dtype=wp.bool, device=self.device))
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}

        self._capturable = (
            cfg.capture
            and self.mdp.backend is mdp2.Backend.WARP
            and self.mdp.is_fully_capturable
            and wp.get_device(self.device).is_cuda
        )
        self._graph = None
        self._stepped = False

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def device(self) -> str:
        return self.sim.device

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def is_captured(self) -> bool:
        """Whether steps replay as one CUDA graph."""
        return self._graph is not None

    @property
    def observations(self) -> VecEnvObs:
        """The observations of every group, as Torch views of buffers that each step overwrites."""
        return {group: values.torch for group, values in self.transition.observations.items()}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Reset every world and return their observations."""
        if seed is not None:
            configure_seed(seed)
        self._reset(self._all_envs)
        return self.observations, self.extras

    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Apply the actions, step the physics, reset the worlds whose episodes ended, and observe."""
        self._step_or_replay(action)
        return self.observations, self.extras

    def close(self) -> None:
        if not self._is_closed:
            self.sim.stop()
            del self.scene
            self.sim.clear_instance()
            self._is_closed = True

    def _step_or_replay(self, action: torch.Tensor) -> None:
        """Run a control step: eagerly the first time, then as a CUDA graph if the MDP allows it."""
        self.actions.torch.copy_(action)

        if self._graph is None and self._capturable and self._stepped:
            # The first step loaded every kernel module, so the capture records launches only.
            self._graph = capture_graph(self.device, self._step)
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._step()

        self._stepped = True
        self.common_step_counter += 1

    def _step(self) -> None:
        # Actions apply before every physics step, as in MDP1, so targets relative to the joint state follow it.
        for _ in range(self.cfg.decimation // self._substeps):
            mdp2.apply_actions(self.mdp, self.state, self.actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt * self._substeps)

        mdp2.evaluate(self.mdp, self.state, self.transition)
        self._reset(self.transition.done)

    def _reset(self, mask: ProxyArray) -> None:
        """Reset the masked worlds, commit their new state to the solver, and observe them."""
        assets = [*self.scene.articulations.values(), *self.scene.rigid_objects.values()]
        for asset in assets:
            asset.reset(env_mask=mask.warp)
        # Clear sensor histories, so that a new episode does not see the contacts that ended the last one.
        for sensor in self.scene.sensors.values():
            sensor.reset(env_mask=mask.warp)

        # Reset events write scene fields in place, which the scene commits to the solver.
        commit = functools.partial(self.scene_data.commit, mask.warp, self.mdp.reset_writes)
        mdp2.reset(self.mdp, self.state, mask, commit)
        commit()

        mdp2.observe(self.mdp, self.state, self.transition.observations, mask)


class MDP2RLEnv(MDP2Env, gym.Env):
    """A Gym environment driven by an MDP2 task, like a :class:`~isaaclab.envs.ManagerBasedRLEnv` without managers.

    :meth:`step` returns copies of the observations after resets, and the rewards and the terminated and truncated
    flags of the step as Torch views of buffers that the next step overwrites. ``extras["log"]`` holds, per term, the
    average over worlds of their last finished episode: the reward sum per second, whether the term ended it, and
    the value that each logged quantity had at its end.
    """

    is_vector_env: ClassVar[bool] = True
    diagnostics: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {"render_modes": [None]}

    cfg: MDP2RLEnvCfg

    def __init__(self, cfg: MDP2RLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg)
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 1 / self.step_dt}

        self.single_observation_space = gym.spaces.Dict(
            {
                group: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))
                for group, dim in self.mdp.observation_dims.items()
            }
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.mdp.action_dim,))
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # Per-term episode statistics, kept on the device so that logging needs no host sync.
        self._reward_names = [op.name.split(".", 1)[1] for op in self.mdp.ops[mdp2.Phase.REWARDS]]
        self._termination_names = [op.name.split(".", 1)[1] for op in self.mdp.ops[mdp2.Phase.TERMINATIONS]]
        # Terms are averaged over the worlds where they are active.
        self._active_worlds = {
            phase: torch.tensor(
                [self.num_envs if op.worlds is None else len(op.worlds) for op in self.mdp.ops[phase]],
                device=self.device,
            )
            for phase in (mdp2.Phase.REWARDS, mdp2.Phase.TERMINATIONS)
        }
        self._episode_rewards = torch.zeros(self.num_envs, len(self._reward_names), device=self.device)
        self._last_episode_rewards = torch.zeros_like(self._episode_rewards)
        self._last_terminations = torch.zeros(self.num_envs, len(self._termination_names), device=self.device)
        self._metrics = {
            op.name.split(".", 1)[1]: (
                self.state.fields[op.writes[0]].torch,
                torch.zeros(self.num_envs, device=self.device),
            )
            for op in self.mdp.ops[mdp2.Phase.QUANTITIES]
            if op.log
        }
        self.obs_buf: VecEnvObs = {}

    @property
    def max_episode_length(self) -> int | float:
        """Control steps until a time-out, from the task's time limit."""
        return self.mdp.max_episode_length or math.inf

    @property
    def max_episode_length_s(self) -> float:
        return self.max_episode_length * self.step_dt

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Control steps since each world's last reset."""
        return self.state.episode_length.torch

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # Copy in place: captured graphs hold the buffer's address.
        self.state.episode_length.torch.copy_(value)

    @staticmethod
    def seed(seed: int = -1) -> int:
        return configure_seed(seed)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Reset every world and return copies of their observations."""
        super().reset(seed, options)
        self._episode_rewards.zero_()

        # Learners keep the observations across the next step, which overwrites the buffers.
        self.obs_buf = {group: values.clone() for group, values in self.observations.items()}
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Apply the actions, step the physics, evaluate the MDP, reset the worlds that ended, and observe."""
        self._step_or_replay(action)
        self._log_episodes()
        self.obs_buf = {group: values.clone() for group, values in self.observations.items()}

        transition = self.transition
        return (
            self.obs_buf,
            transition.rewards.torch,
            transition.terminated.torch,
            transition.truncated.torch,
            self.extras,
        )

    def _log_episodes(self) -> None:
        """Add the step's term rewards to the episode sums, and record the episodes that ended in ``extras["log"]``."""
        transition = self.transition
        ended = transition.done.torch[:, None]
        self._episode_rewards += transition.reward_terms.torch

        self._last_episode_rewards = torch.where(ended, self._episode_rewards, self._last_episode_rewards)
        terminations = transition.termination_terms.torch.float()
        self._last_terminations = torch.where(ended, terminations, self._last_terminations)
        self._episode_rewards = torch.where(ended, 0.0, self._episode_rewards)

        rewards = self._last_episode_rewards.sum(dim=0) / self._active_worlds[mdp2.Phase.REWARDS]
        rewards /= self.max_episode_length_s
        terminations = self._last_terminations.sum(dim=0) / self._active_worlds[mdp2.Phase.TERMINATIONS]
        log = {f"Episode_Reward/{name}": rewards[k] for k, name in enumerate(self._reward_names)}
        log |= {f"Episode_Termination/{name}": terminations[k] for k, name in enumerate(self._termination_names)}
        # Quantities keep the values of the step that ended the episode until the next step.
        for name, (value, last) in self._metrics.items():
            last.copy_(torch.where(transition.done.torch, value, last))
            log[f"Metrics/{name}"] = last.mean()
        self.extras["log"] = log
