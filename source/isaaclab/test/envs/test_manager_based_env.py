# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

from unittest.mock import Mock

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg, make_empty_manager_based_rl_env_cfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

pytestmark = pytest.mark.integration


def dummy_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Return a dummy observation."""
    return torch.randn((env.num_envs, 1), device=env.device)


@configclass
class EmptyObservationWithHistoryCfg:
    """Empty observation with history specifications for the environment."""

    @configclass
    class EmptyObservationGroupWithHistoryCfg(ObsGroup):
        """Empty observation with history specifications for the environment."""

        dummy_term: ObsTerm = ObsTerm(func=dummy_observation)

        def __post_init__(self):
            self.history_length = 5

    empty_observation: EmptyObservationGroupWithHistoryCfg = EmptyObservationGroupWithHistoryCfg()


def episode_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose per-environment step counters as a noiseless observation."""
    return env.episode_length_buf[:, None].float()


# Both devices, so the CPU and GPU simulation pipelines each build and step an environment.
@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
@pytest.mark.parametrize(
    "env_type,cfg_factory",
    [(ManagerBasedEnv, make_empty_manager_based_env_cfg), (ManagerBasedRLEnv, make_empty_manager_based_rl_env_cfg)],
)
def test_step_updates_observation_history(device, env_type, cfg_factory, monkeypatch):
    """Real steps advance history, and reading a noiseless group preserves policy RNG."""
    # create a new stage
    sim_utils.create_new_stage()
    # create environment with history length of 5
    cfg = cfg_factory(device=device, num_envs=3)
    cfg.observations = EmptyObservationWithHistoryCfg()
    captured = None

    def sample_terminal_observation(env):
        if captured is not None:
            captured["latest"] = env.episode_length_buf.clone()
            captured["observation"] = env.observation_manager.compute_group("state", update_history=False)
        return torch.zeros(env.num_envs, device=env.device)

    if env_type is ManagerBasedRLEnv:
        cfg.observations.empty_observation.enable_corruption = True
        cfg.observations.empty_observation.dummy_term = ObsTerm(
            func=episode_step, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1)
        )
        cfg.observations.state = ObsGroup(enable_corruption=False)
        cfg.observations.state.step = ObsTerm(func=episode_step)
        cfg.rewards.sample = RewardTermCfg(func=sample_terminal_observation, weight=1.0)
        cfg.terminations.stop = TerminationTermCfg(
            func=lambda env: (env.episode_length_buf == 4) & (torch.arange(env.num_envs, device=env.device) == 0)
        )
    env = env_type(cfg=cfg)
    assert env.action_manager.total_action_dim == 0
    history = env.observation_manager._group_obs_term_history_buffer["empty_observation"]["dummy_term"]
    reset_scene = Mock(wraps=env.scene.reset)
    monkeypatch.setattr(env.scene, "reset", reset_scene)
    reset_history = Mock(wraps=history.reset)
    monkeypatch.setattr(history, "reset", reset_history)

    torch.testing.assert_close(
        history.current_length,
        torch.zeros((env.num_envs,), device=device, dtype=torch.int64),
    )

    # step the environment repeatedly and verify that history advances with each step
    for num_steps in (1, 2):
        env.step(torch.randn_like(env.action_manager.action))
        torch.testing.assert_close(
            history.current_length,
            torch.full((env.num_envs,), num_steps, device=device, dtype=torch.int64),
        )

    # Preserve the slice through to buffer storage, avoiding advanced-indexed resets.
    selected = slice(1, None, 2)
    env.reset(env_ids=selected)
    assert reset_history.call_args.kwargs["batch_ids"] is selected
    assert reset_scene.call_args.args[0] is selected
    torch.testing.assert_close(history.current_length, torch.tensor([3, 1, 3], device=device))
    env.reset_to({}, env_ids=slice(0, None, 2))
    torch.testing.assert_close(history.current_length, torch.tensor([1, 2, 1], device=device))
    env.reset()
    assert reset_history.call_args.kwargs["batch_ids"] == slice(None)
    torch.testing.assert_close(history.current_length, torch.ones_like(history.current_length))
    env.reset(slice(None), seed=42)
    env.reset_to({})
    torch.testing.assert_close(history.current_length, torch.ones_like(history.current_length))
    # Explicit None keeps selecting all environments, normalized to the full slice.
    env.step(torch.randn_like(env.action_manager.action))
    env.reset(None)
    assert reset_scene.call_args.args[0] == slice(None)
    env.step(torch.randn_like(env.action_manager.action))
    env.reset_to({}, None)
    assert reset_history.call_args.kwargs["batch_ids"] == slice(None)
    torch.testing.assert_close(history.current_length, torch.ones_like(history.current_length))
    assert not hasattr(env.scene, "resolve_env_ids")

    if env_type is ManagerBasedRLEnv:
        outputs, rng_states = [], []
        action = torch.empty_like(env.action_manager.action)
        for capture in (False, True):
            captured = {} if capture else None
            env.reset(seed=17)
            start_time = env.sim.backend_manager.get_simulation_time()
            trajectory = [env.step(action)[0]["empty_observation"].clone() for _ in range(4)]
            elapsed = env.sim.backend_manager.get_simulation_time() - start_time
            assert elapsed == pytest.approx(4 * env.step_dt, rel=1e-5, abs=1e-8)
            outputs.append(torch.stack(trajectory))
            rng = torch.get_rng_state()
            if device.startswith("cuda"):
                rng = torch.cat((rng, torch.cuda.get_rng_state(device)))
            rng_states.append(rng)
            if capture:
                assert torch.equal(captured["latest"], torch.full_like(captured["latest"], 4))
                assert torch.equal(captured["observation"], captured["latest"][:, None].float())
            assert env.episode_length_buf.tolist() == [0, 4, 4]
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(rng_states[0], rng_states[1])
        # Continuing environments retain the initial observation and steps 1–4; reset history contains only step zero.
        assert torch.all((outputs[1][-1, 1:] - torch.arange(5, device=device)).abs() <= 0.1)
        assert torch.all(outputs[1][-1, 0].abs() <= 0.1)
    env.close()
