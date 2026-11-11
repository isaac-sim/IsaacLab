# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environments that run an MDP2 task on an interactive scene, in place of the managers of manager-based ones."""

from __future__ import annotations

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
        cfg, sim = self.cfg, self.sim
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
        builder = mdp2.MDPBuilder(self.scene_data.schema)
        builder.add_task(cfg.task)
        self.mdp = builder.finalize(device=self.device, backend=cfg.backend)
        self.state = self.mdp.state(self.scene_data, seed=cfg.seed or 0)
        self.transition = self.mdp.transition()
        self.actions = ProxyArray(wp.zeros((self.num_envs, self.mdp.action_dim), dtype=wp.float32, device=self.device))
        self._all_envs = ProxyArray(wp.ones(self.num_envs, dtype=wp.bool, device=self.device))
        self._capturable = (
            cfg.capture
            and cfg.backend is mdp2.Backend.WARP
            and self.mdp.is_fully_capturable
            and wp.get_device(self.device).is_cuda
        )
        self._graph = None
        self._stepped = False
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}

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
        mdp2.apply_actions(self.mdp, self.state, self.actions)
        for _ in range(self.cfg.decimation // self._substeps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt * self._substeps)
        mdp2.evaluate(self.mdp, self.state, self.transition)
        self._reset(self.transition.done)

    def _reset(self, mask: ProxyArray) -> None:
        """Reset the masked worlds, commit their new state to the solver, and observe them."""
        for articulation in self.scene.articulations.values():
            articulation.reset(env_mask=mask.warp)
        mdp2.reset(self.mdp, self.state, mask)
        for articulation in self.scene.articulations.values():
            articulation_ids = articulation.root_view.articulation_ids
            self.sim.backend_manager.invalidate_fk(env_mask=mask.warp, articulation_ids=articulation_ids)
        self.sim.forward()
        mdp2.observe(self.mdp, self.state, self.transition.observations, mask)


class MDP2RLEnv(MDP2Env, gym.Env):
    """A Gym environment driven by an MDP2 task, like a :class:`~isaaclab.envs.ManagerBasedRLEnv` without managers.

    :meth:`step` returns the observations after resets, the rewards, and the terminated and truncated flags of the
    step, as Torch views of buffers that the next step overwrites.
    """

    is_vector_env: ClassVar[bool] = True
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

    @property
    def max_episode_length(self) -> int:
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

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Apply the actions, step the physics, evaluate the MDP, reset the worlds that ended, and observe."""
        self._step_or_replay(action)
        transition = self.transition
        return (
            self.observations,
            transition.rewards.torch,
            transition.terminated.torch,
            transition.truncated.torch,
            self.extras,
        )
