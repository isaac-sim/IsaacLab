# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based fixed-agent multi-agent reinforcement-learning environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)

from .common import ActionType, EnvStepReturn, ObsType, StateType
from .manager_based_env import ManagerBasedEnv
from .manager_based_marl_env_cfg import ManagerBasedMARLEnvCfg


class ManagerBasedMARLEnv(ManagerBasedEnv, gym.Env):
    """The fixed-agent manager-based workflow multi-agent RL environment.

    The agent identifiers and order come from the insertion order of
    :attr:`ManagerBasedMARLEnvCfg.agents` and remain fixed for the lifetime of
    the environment. Actions, observations, rewards, and termination signals
    are dictionaries keyed by those identifiers. A centralized state is
    available only when :attr:`ManagerBasedMARLEnvCfg.state` explicitly
    configures one observation manager with one output group.

    Agent-local terms receive an :class:`Agent` facade, so existing manager terms can
    continue to use ``env.action_manager`` and related manager properties. Shared terms
    receive this environment. They must explicitly select an agent with :meth:`get_agent`
    before accessing an agent-local manager.
    """

    is_vector_env: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
    }

    cfg: ManagerBasedMARLEnvCfg

    class Agent:
        """A view of one agent and its local managers."""

        def __init__(self, agent_id: str, agent_cfg: ManagerBasedMARLEnvCfg.AgentCfg, parent: ManagerBasedMARLEnv):
            self.agent_id = agent_id
            self.agent_cfg = agent_cfg
            self.parent = parent

        @property
        def action_manager(self) -> ActionManager:
            """The agent-local action manager."""
            return self.parent.action_managers[self.agent_id]

        @property
        def observation_manager(self) -> ObservationManager:
            """The agent-local observation manager."""
            return self.parent.observation_managers[self.agent_id]

        @property
        def reward_manager(self) -> RewardManager:
            """The agent-local reward manager."""
            return self.parent.reward_managers[self.agent_id]

        @property
        def termination_manager(self) -> TerminationManager:
            """The agent-local termination manager."""
            return self.parent.termination_managers[self.agent_id]

        @property
        def extras(self) -> dict:
            """The agent-local extras dictionary."""
            return self.parent.extras[self.agent_id]

        def __getattr__(self, name: str) -> Any:
            """Proxy shared environment properties to the parent environment."""
            return getattr(self.parent, name)

    def __init__(self, cfg: ManagerBasedMARLEnvCfg, render_mode: str | None = None, **kwargs):
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        super().__init__(cfg=cfg)
        self.render_mode = render_mode
        self.metadata["render_fps"] = 1 / self.step_dt
        self.has_rtx_sensors = self.sim.get_setting("/isaaclab/render/rtx_sensors")
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        print("[INFO]: Completed setting up the environment...")

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length [s]."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def num_agents(self) -> int:
        """Number of fixed agents."""
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Number of all possible agents."""
        return len(self.possible_agents)

    @property
    def action_manager(self) -> ActionManager:
        raise ValueError(self._ambiguous_manager_message("action_manager"))

    @property
    def observation_manager(self) -> ObservationManager:
        raise ValueError(self._ambiguous_manager_message("observation_manager"))

    @property
    def reward_manager(self) -> RewardManager:
        raise ValueError(self._ambiguous_manager_message("reward_manager"))

    @property
    def termination_manager(self) -> TerminationManager:
        raise ValueError(self._ambiguous_manager_message("termination_manager"))

    def _ambiguous_manager_message(self, manager_name: str) -> str:
        return (
            f"ManagerBasedMARLEnv has no singular {manager_name}. Select an agent with "
            "env.get_agent(agent_id) or use a custom term that explicitly selects the intended agent."
        )

    def get_agent(self, agent_id: str) -> Agent:
        """Return the fixed agent with the given identifier."""
        try:
            return self._agents[agent_id]
        except KeyError as error:
            raise KeyError(f"Unknown agent '{agent_id}'. Expected one of: {list(self.possible_agents)}.") from error

    def observation_space(self, agent: str) -> gym.Space:
        """Return the unbatched observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """Return the unbatched action space for an agent."""
        return self.action_spaces[agent]

    def load_managers(self) -> None:
        """Load shared and agent-local managers in dependency order."""
        print("[INFO] Event Manager: ", self.event_manager)
        from isaaclab.managers import RecorderManager

        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        self.possible_agents = list(self.cfg.agents)
        self.agents = list(self.possible_agents)
        self._agents = {
            agent_id: self.Agent(agent_id, agent_cfg, self) for agent_id, agent_cfg in self.cfg.agents.items()
        }
        self.action_managers: dict[str, ActionManager] = {}
        self.observation_managers: dict[str, ObservationManager] = {}
        self.termination_managers: dict[str, TerminationManager] = {}
        self.reward_managers: dict[str, RewardManager] = {}

        for agent_id, agent in self._agents.items():
            self.action_managers[agent_id] = ActionManager(agent.agent_cfg.actions, agent)
            print(f"[INFO] Action Manager ({agent_id}): ", self.action_managers[agent_id])
        for agent_id, agent in self._agents.items():
            self.observation_managers[agent_id] = ObservationManager(agent.agent_cfg.observations, agent)
            print(f"[INFO] Observation Manager ({agent_id}): ", self.observation_managers[agent_id])
        for agent_id, agent in self._agents.items():
            self.termination_managers[agent_id] = TerminationManager(agent.agent_cfg.terminations, agent)
            print(f"[INFO] Termination Manager ({agent_id}): ", self.termination_managers[agent_id])
        for agent_id, agent in self._agents.items():
            self.reward_managers[agent_id] = RewardManager(agent.agent_cfg.rewards, agent)
            print(f"[INFO] Reward Manager ({agent_id}): ", self.reward_managers[agent_id])

        self.state_manager = ObservationManager(self.cfg.state, self) if self.cfg.state is not None else None
        if self.state_manager is not None:
            print("[INFO] State Observation Manager: ", self.state_manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()
        self.extras = {agent_id: {} for agent_id in self.possible_agents}
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self) -> None:
        """Register shared and agent-local manager plots without ambiguous aliases."""
        if not self.sim.has_gui and not self.sim.has_active_visualizers():
            self.manager_visualizers = {}
            return
        managers = {
            "command_manager": self.command_manager,
            "curriculum_manager": self.curriculum_manager,
            **{f"{agent_id}.action_manager": manager for agent_id, manager in self.action_managers.items()},
            **{f"{agent_id}.observation_manager": manager for agent_id, manager in self.observation_managers.items()},
            **{f"{agent_id}.reward_manager": manager for agent_id, manager in self.reward_managers.items()},
            **{f"{agent_id}.termination_manager": manager for agent_id, manager in self.termination_managers.items()},
        }
        if self.state_manager is not None:
            managers["state_manager"] = self.state_manager
        for visualizer in self.sim.visualizers:
            visualizer.add_live_plots(managers)
        self.manager_visualizers = {
            name: manager
            for visualizer in self.sim.visualizers
            for name, manager in getattr(visualizer, "kit_manager_visualizers", {}).items()
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, ObsType], dict]:
        """Reset all sub-environments and return per-agent observations."""
        if seed is not None:
            self.seed(seed)
        env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.recorder_manager.record_pre_reset(env_ids)
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        self.recorder_manager.record_post_reset(env_ids)
        self.obs_dict = self._get_observations(update_history=True)
        if self.cfg.wait_for_textures and self.has_rtx_sensors and hasattr(self.sim.physics_manager, "assets_loading"):
            while self.sim.physics_manager.assets_loading():
                self.sim.render()
        return self.obs_dict, self.extras

    def step(self, actions: dict[str, ActionType]) -> EnvStepReturn:
        """Advance the simulation once using one action tensor for every fixed agent."""
        expected = set(self.possible_agents)
        supplied = set(actions)
        if expected != supplied:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise ValueError(f"Agent actions must match fixed agents; missing={missing}, unexpected={unexpected}.")
        for agent_id in self.possible_agents:
            action = actions[agent_id]
            if (
                not isinstance(action, torch.Tensor)
                or action.ndim != 2
                or action.shape[0] != self.num_envs
                or action.shape[1] != self.action_managers[agent_id].total_action_dim
            ):
                received = getattr(action, "shape", None)
                raise ValueError(
                    f"Invalid action shape for agent '{agent_id}', expected "
                    f"({self.num_envs}, {self.action_managers[agent_id].total_action_dim}), received {received}."
                )
            self.action_managers[agent_id].process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.is_rendering
        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            for manager in self.action_managers.values():
                manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                for manager in self.action_managers.values():
                    manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.terminated_dict = {agent_id: manager.compute() for agent_id, manager in self.termination_managers.items()}
        self.time_out_dict = {agent_id: manager.time_outs for agent_id, manager in self.termination_managers.items()}
        self.reset_buf[:] = math.prod(self.terminated_dict.values()) | math.prod(self.time_out_dict.values())
        self.reward_dict = {
            agent_id: manager.compute(dt=self.step_dt) for agent_id, manager in self.reward_managers.items()
        }

        if self.recorder_manager.active_terms:
            self.obs_dict = self._get_observations()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            if self.cfg.compute_final_obs:
                for agent_id, observation in self._get_observations().items():
                    self.extras[agent_id]["final_obs"] = observation
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        if self.sim.consume_reset_request():
            not_yet_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            not_yet_reset[reset_env_ids] = False
            manual_reset_ids = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
            if len(manual_reset_ids) > 0:
                for terminated in self.terminated_dict.values():
                    terminated[manual_reset_ids] = True
                self.recorder_manager.record_pre_reset(manual_reset_ids)
                self._reset_idx(manual_reset_ids)
                self.recorder_manager.record_post_reset(manual_reset_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        for recorder in self.video_recorders:
            recorder.step()
        self.obs_dict = self._get_observations(update_history=True)
        return self.obs_dict, self.reward_dict, self.terminated_dict, self.time_out_dict, self.extras

    def state(self) -> StateType | None:
        """Return the centralized state, or None when no state manager is configured."""
        if self.state_manager is None:
            return None
        return self._unwrap_single_observation(self.state_manager, "state", update_history=False)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without advancing the physics simulation."""
        if not self.has_rtx_sensors and not recompute:
            self.sim.render()
        if self.render_mode == "rgb_array":
            import warnings

            warnings.warn(
                "render_mode='rgb_array' is deprecated and will be removed in a future release. "
                "Use VideoRecorderCfg on env_cfg.video_recorders to capture frames instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return None
        if self.render_mode in ("human", None):
            return None
        raise NotImplementedError(
            f"Render mode '{self.render_mode}' is not supported. Use {self.metadata['render_modes']}."
        )

    def close(self) -> None:
        """Release local managers before shared simulation resources."""
        if self._is_closed:
            return
        self.sim.stop()
        if isinstance(getattr(self, "obs_dict", None), dict):
            self.obs_dict.clear()
        for recorder in getattr(self, "video_recorders", []):
            recorder.close()
        for managers in (
            getattr(self, "reward_managers", {}).values(),
            getattr(self, "termination_managers", {}).values(),
            getattr(self, "observation_managers", {}).values(),
            getattr(self, "action_managers", {}).values(),
        ):
            for manager in managers:
                del manager
        self.reward_managers.clear()
        self.termination_managers.clear()
        self.observation_managers.clear()
        self.action_managers.clear()
        self._agents.clear()
        if getattr(self, "state_manager", None) is not None:
            del self.state_manager
        del self.curriculum_manager
        del self.command_manager
        del self.recorder_manager
        del self.event_manager
        del self.scene
        self.sim.clear_instance()
        self.observation_spaces = None
        self.action_spaces = None
        self.state_space = None
        if self._window is not None:
            self._window = None
        self._is_closed = True

    def _get_observations(self, update_history: bool = False) -> dict[str, ObsType]:
        return {
            agent_id: self._unwrap_single_observation(manager, agent_id, update_history)
            for agent_id, manager in self.observation_managers.items()
        }

    @staticmethod
    def _unwrap_single_observation(
        manager: ObservationManager, owner: str, update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        observations = manager.compute(update_history=update_history)
        if len(observations) != 1:
            raise ValueError(
                f"Observation manager for '{owner}' must define exactly one output group, got {list(observations)}."
            )
        return next(iter(observations.values()))

    def _configure_gym_env_spaces(self) -> None:
        self.observation_spaces = {
            agent_id: self._single_observation_space(manager, agent_id)
            for agent_id, manager in self.observation_managers.items()
        }
        self.action_spaces = {
            agent_id: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(manager.total_action_dim,))
            for agent_id, manager in self.action_managers.items()
        }
        self.state_space = (
            None if self.state_manager is None else self._single_observation_space(self.state_manager, "state")
        )

    @staticmethod
    def _single_observation_space(manager: ObservationManager, owner: str) -> gym.Space:
        if len(manager.active_terms) != 1:
            raise ValueError(
                f"Observation manager for '{owner}' must define exactly one output group, "
                f"got {list(manager.active_terms)}."
            )
        group_name = next(iter(manager.active_terms))
        group_dim = manager.group_obs_dim[group_name]
        if manager.group_obs_concatenate[group_name]:
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
        term_cfgs = manager._group_obs_term_cfgs[group_name]
        terms = {}
        for term_name, term_dim, term_cfg in zip(manager.active_terms[group_name], group_dim, term_cfgs):
            terms[term_name] = gym.spaces.Box(
                low=-np.inf if term_cfg.clip is None else term_cfg.clip[0],
                high=np.inf if term_cfg.clip is None else term_cfg.clip[1],
                shape=term_dim,
            )
        return gym.spaces.Dict(terms)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        self.curriculum_manager.compute(env_ids=env_ids)
        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=self._sim_step_counter // self.cfg.decimation
            )
        self.extras["log"] = {}
        for agent_id in self.possible_agents:
            self.extras[agent_id]["log"] = {}
            log = self.extras[agent_id]["log"]
            log.update(self.observation_managers[agent_id].reset(env_ids))
            log.update(self.action_managers[agent_id].reset(env_ids))
            log.update(self.reward_managers[agent_id].reset(env_ids))
            log.update(self.termination_managers[agent_id].reset(env_ids))
        if self.state_manager is not None:
            self.extras["log"].update(self.state_manager.reset(env_ids))
        self.extras["log"].update(self.curriculum_manager.reset(env_ids))
        self.extras["log"].update(self.command_manager.reset(env_ids))
        self.extras["log"].update(self.event_manager.reset(env_ids))
        self.extras["log"].update(self.recorder_manager.reset(env_ids))
        self.episode_length_buf[env_ids] = 0
        self.sim.render_context.reset_scene_state_cadence()
