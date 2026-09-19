# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for reinforcement learning environment wrappers."""

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.integration

_NUM_ENVS = 2
_EPISODE_STEPS = 3


def _wrap_env(library: str, env: Any) -> Any:
    if library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        return RslRlVecEnvWrapper(env)
    if library == "rl_games":
        from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

        return RlGamesVecEnvWrapper(env, "cuda:0", 100, 100)
    if library == "sb3":
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper

        return Sb3VecEnvWrapper(env)
    if library == "skrl":
        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        return SkrlVecEnvWrapper(env)
    raise ValueError(f"Unsupported RL library: {library}")


@pytest.fixture
def raw_env(task: str, finite_horizon: bool) -> Iterator[Any]:
    import gymnasium as gym
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg(task, device="cuda:0", num_envs=_NUM_ENVS)
    cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    cfg.episode_length_s = _EPISODE_STEPS * cfg.decimation * cfg.sim.dt
    cfg.seed = 42
    cfg.is_finite_horizon = finite_horizon
    with launch_simulation(cfg, {"headless": True, "visualizer": None, "visualizer_explicit": True}):
        env = gym.make(task, cfg=cfg)
        try:
            yield env
        finally:
            env.close()


@pytest.mark.parametrize(
    ("library", "finite_horizon"),
    [("rsl_rl", False), ("rsl_rl", True), ("rl_games", False), ("sb3", False), ("skrl", False)],
)
@pytest.mark.parametrize("task", ["Isaac-Cartpole", "Isaac-Cartpole-Direct"])
def test_wrapper_reset_step_and_timeout(library: str, finite_horizon: bool, raw_env: Any) -> None:
    env = _wrap_env(library, raw_env)
    _assert_finite(env.reset())
    if library == "rsl_rl":
        _assert_observation_buffer(env)
    saw_done = False
    with torch.inference_mode():
        for _ in range(_EPISODE_STEPS + 1):
            actions = torch.zeros((_NUM_ENVS, 1), device=raw_env.unwrapped.device)
            transition = env.step(actions.cpu().numpy() if library == "sb3" else actions)
            _assert_finite(transition)
            _, rewards, dones, *extras = transition
            assert rewards.shape[0] == dones.shape[0] == _NUM_ENVS
            if library == "skrl":
                dones = dones | extras[0]
            saw_done |= bool(dones.any())
            if library == "rsl_rl":
                _assert_observation_buffer(env)
                assert ("time_outs" in extras[0]) is not finite_horizon
                if not finite_horizon:
                    torch.testing.assert_close(extras[0]["time_outs"], raw_env.unwrapped.reset_time_outs)
            elif library == "sb3" and bool(dones.any()):
                for index in np.flatnonzero(dones):
                    assert extras[0][index]["terminal_observation"] is not None
    assert saw_done, "The short episode must exercise automatic reset"


def _assert_observation_buffer(env: Any) -> None:
    observations = env.get_observations()
    assert isinstance(observations, TensorDict)
    assert set(observations.keys()) == set(env.unwrapped.obs_buf)
    for key, value in env.unwrapped.obs_buf.items():
        torch.testing.assert_close(observations[key], value)


def _assert_finite(data: Any) -> None:
    if isinstance(data, torch.Tensor):
        assert torch.isfinite(data).all()
    elif isinstance(data, np.ndarray):
        assert np.isfinite(data).all()
    elif isinstance(data, (Mapping, TensorDict)):
        for value in data.values():
            _assert_finite(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            _assert_finite(value)
