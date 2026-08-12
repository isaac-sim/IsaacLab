# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime tests for the manager-based MARL environment using an empty scene."""

from __future__ import annotations

from isaaclab.app import AppLauncher

# Launch the simulator before importing Isaac Lab modules that require Kit.
simulation_app = AppLauncher(headless=True).app

import gymnasium as gym
import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedMARLEnv, ManagerBasedMARLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import PreStepActionsRecorderCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    RecorderManagerBaseCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.test.env_cfgs import EmptyManagerCfg, EmptySceneCfg
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.integration


def _observation(env: ManagerBasedMARLEnv.Agent) -> torch.Tensor:
    """Return a minimal agent-local observation."""
    return torch.full((env.num_envs, 1), float(env.episode_length_buf[0]), device=env.device)


def _termination(env: ManagerBasedMARLEnv.Agent, done: bool) -> torch.Tensor:
    """Return an optional terminal signal while recording manager ordering."""
    env._trace.append(f"termination:{env.agent_id}")
    return torch.full((env.num_envs,), done, device=env.device, dtype=torch.bool)


def _reward(env: ManagerBasedMARLEnv.Agent) -> torch.Tensor:
    """Return a minimal reward while recording manager ordering."""
    env._trace.append(f"reward:{env.agent_id}")
    return torch.ones(env.num_envs, device=env.device)


@configclass
class _ObservationsCfg:
    """Minimal one-group observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        value = ObsTerm(func=_observation)

    policy: PolicyCfg = PolicyCfg()


@configclass
class _RewardsCfg:
    """Minimal reward configuration."""

    reward = RewardTermCfg(func=_reward, weight=1.0)


@configclass
class _TerminationsCfg:
    """Minimal termination configuration."""

    done = TerminationTermCfg(func=_termination, params={"done": False})


@configclass
class _AmbiguousRecorderCfg(RecorderManagerBaseCfg):
    """Recorder configuration that deliberately requests a singular action manager."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    actions = PreStepActionsRecorderCfg()


@configclass
class _AmbiguousCurriculumCfg:
    """Curriculum configuration that deliberately requests a singular reward manager."""

    reward_weight = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={"term_name": "reward", "weight": 1.0, "num_steps": 1},
    )


def _make_cfg(
    *, compute_final_obs: bool = False, recorders: object | None = None, curriculum: object | None = None
) -> ManagerBasedMARLEnvCfg:
    """Create a CPU empty-scene manager-based MARL configuration."""
    return ManagerBasedMARLEnvCfg(
        decimation=1,
        episode_length_s=1.0,
        sim=sim_utils.SimulationCfg(device="cpu", dt=0.01, render_interval=1),
        scene=EmptySceneCfg(num_envs=3, env_spacing=1.0),
        agents={
            "left": ManagerBasedMARLEnvCfg.AgentCfg(
                actions=EmptyManagerCfg(),
                observations=_ObservationsCfg(),
                rewards=_RewardsCfg(),
                terminations=_TerminationsCfg(),
            ),
            "right": ManagerBasedMARLEnvCfg.AgentCfg(
                actions=EmptyManagerCfg(),
                observations=_ObservationsCfg(),
                rewards=_RewardsCfg(),
                terminations=_TerminationsCfg(),
            ),
        },
        recorders=RecorderManagerBaseCfg() if recorders is None else recorders,
        curriculum=curriculum,
        compute_final_obs=compute_final_obs,
    )


@pytest.fixture
def env() -> ManagerBasedMARLEnv:
    """Create and close an empty-scene MARL environment for each test."""
    sim_utils.create_new_stage()
    environment = ManagerBasedMARLEnv(_make_cfg())
    environment._trace = []
    yield environment
    environment.close()


def _actions(env: ManagerBasedMARLEnv, batch_size: int | None = None) -> dict[str, torch.Tensor]:
    """Return empty actions using the requested leading batch dimension."""
    if batch_size is None:
        batch_size = env.num_envs
    return {agent: torch.zeros(batch_size, 0, device=env.device) for agent in env.possible_agents}


def test_initialization_exposes_fixed_agents_and_spaces(env: ManagerBasedMARLEnv):
    """The real environment exposes fixed agents, local managers, and unwrapped spaces."""
    assert env.agents == ["left", "right"]
    assert env.possible_agents == ["left", "right"]
    assert env.num_agents == 2
    assert isinstance(env.observation_space("left"), gym.spaces.Box)
    assert env.action_space("right").shape == (0,)
    assert env.get_agent("left").extras is env.extras["left"]


@pytest.mark.parametrize("batch_size", [2, 1])
def test_step_rejects_invalid_agent_action_batch_size(env: ManagerBasedMARLEnv, batch_size: int):
    """Action tensors must have exactly one row per vectorized sub-environment."""
    with pytest.raises(ValueError, match="Invalid action shape for agent 'left'"):
        env.step(_actions(env, batch_size=batch_size))


def test_step_rejects_missing_and_extra_agent_actions(env: ManagerBasedMARLEnv):
    """Action dictionaries must exactly match the fixed agent identifiers."""
    with pytest.raises(ValueError, match=r"missing=\['right'\].*unexpected=\['extra'\]"):
        env.step({"left": torch.zeros(env.num_envs, 0), "extra": torch.zeros(env.num_envs, 0)})


def test_step_returns_per_agent_rewards_and_dones_in_manager_order(
    env: ManagerBasedMARLEnv, monkeypatch: pytest.MonkeyPatch
):
    """A real step processes every action before all dones and then all rewards."""
    for agent_id, manager in env.action_managers.items():
        process_action = manager.process_action
        apply_action = manager.apply_action
        monkeypatch.setattr(
            manager,
            "process_action",
            lambda action, process_action=process_action, agent_id=agent_id: (
                env._trace.append(f"process:{agent_id}"),
                process_action(action),
            )[1],
        )
        monkeypatch.setattr(
            manager,
            "apply_action",
            lambda apply_action=apply_action, agent_id=agent_id: (
                env._trace.append(f"apply:{agent_id}"),
                apply_action(),
            )[1],
        )

    observations, rewards, terminated, truncated, _ = env.step(_actions(env))

    assert list(observations) == env.possible_agents
    assert list(rewards) == env.possible_agents
    assert list(terminated) == env.possible_agents
    assert list(truncated) == env.possible_agents
    assert max(env._trace.index(f"termination:{agent}") for agent in env.possible_agents) < min(
        env._trace.index(f"reward:{agent}") for agent in env.possible_agents
    )


def test_reset_synchronizes_scene_and_replaces_agent_logs(env: ManagerBasedMARLEnv, monkeypatch: pytest.MonkeyPatch):
    """Explicit reset synchronizes scene state and clears stale per-agent metrics."""
    calls: list[str] = []
    write_data_to_sim = env.scene.write_data_to_sim
    forward = env.sim.forward
    monkeypatch.setattr(env.scene, "write_data_to_sim", lambda: (calls.append("write"), write_data_to_sim())[1])
    monkeypatch.setattr(env.sim, "forward", lambda: (calls.append("forward"), forward())[1])
    reward_reset = env.reward_managers["left"].reset
    reset_count = 0

    def reset_with_one_transient_metric(env_ids):
        nonlocal reset_count
        reset_count += 1
        metrics = reward_reset(env_ids)
        if reset_count == 1:
            metrics["transient"] = 1.0
        return metrics

    monkeypatch.setattr(env.reward_managers["left"], "reset", reset_with_one_transient_metric)
    env.reset()
    assert calls == ["write", "forward"]
    assert env.extras["left"]["log"]["transient"] == 1.0
    env.reset()
    assert "transient" not in env.extras["left"]["log"]


def test_state_returns_none_when_disabled(env: ManagerBasedMARLEnv):
    """Centralized state is absent when no state manager is configured."""
    assert env.state() is None


def test_state_returns_configured_group():
    """Centralized state uses its configured one-group observation manager."""
    cfg = _make_cfg()
    cfg.state = _ObservationsCfg()
    sim_utils.create_new_stage()
    state_env = ManagerBasedMARLEnv(cfg)
    state_env._trace = []
    assert isinstance(state_env.state(), torch.Tensor)
    assert state_env.state().shape == (state_env.num_envs, 1)
    state_env.close()


def test_compute_final_obs_is_namespaced_per_agent_after_autoreset():
    """Same-step autoreset captures one terminal observation payload per agent."""
    cfg = _make_cfg(compute_final_obs=True)
    cfg.agents["left"].terminations.done.params["done"] = True
    cfg.agents["right"].terminations.done.params["done"] = True
    sim_utils.create_new_stage()
    env = ManagerBasedMARLEnv(cfg)
    env._trace = []
    env.step(_actions(env))
    assert "final_obs" in env.extras["left"]
    assert "final_obs" in env.extras["right"]
    env.close()


def test_ambiguous_builtin_recorder_does_not_select_first_agent():
    """Built-in recorder terms fail with guidance instead of choosing an agent."""
    sim_utils.create_new_stage()
    env = ManagerBasedMARLEnv(_make_cfg(recorders=_AmbiguousRecorderCfg()))
    env._trace = []
    with pytest.raises(ValueError, match="get_agent"):
        env.step(_actions(env))
    env.close()


def test_ambiguous_builtin_curriculum_does_not_select_first_agent():
    """Built-in curricula fail during manager setup with agent-selection guidance."""
    sim_utils.create_new_stage()
    with pytest.raises(ValueError, match="get_agent"):
        ManagerBasedMARLEnv(_make_cfg(curriculum=_AmbiguousCurriculumCfg()))


def test_export_io_descriptors_is_rejected_before_environment_initialization():
    """MARL rejects inherited singular-manager IO descriptor export explicitly."""
    cfg = _make_cfg()
    cfg.export_io_descriptors = True
    with pytest.raises(ValueError, match="does not support export_io_descriptors"):
        cfg.validate()
