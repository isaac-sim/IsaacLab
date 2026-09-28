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
    if library == "torchrl":
        pytest.importorskip("torchrl")
        from isaaclab_rl.torchrl import IsaacLabTorchRLWrapper

        return IsaacLabTorchRLWrapper(env)
    raise ValueError(f"Unsupported RL library: {library}")


@pytest.fixture
def raw_env(task: str, library: str, finite_horizon: bool) -> Iterator[Any]:
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
    cfg.compute_final_obs = library == "torchrl"
    with launch_simulation(cfg, {"headless": True, "visualizer": None, "visualizer_explicit": True}):
        env = gym.make(task, cfg=cfg)
        try:
            yield env
        finally:
            env.close()


# Only the RSL-RL wrapper branches on manager-based vs direct envs, so the task rotates across rows
# instead of crossing every library.
@pytest.mark.parametrize(
    ("library", "finite_horizon", "task"),
    [
        ("rsl_rl", False, "Isaac-Cartpole"),
        ("rsl_rl", True, "Isaac-Cartpole-Direct"),
        ("rl_games", False, "Isaac-Cartpole"),
        ("sb3", False, "Isaac-Cartpole"),
        ("skrl", False, "Isaac-Cartpole-Direct"),
        ("torchrl", False, "Isaac-Cartpole"),
        ("torchrl", True, "Isaac-Cartpole-Direct"),
    ],
)
def test_wrapper_reset_step_and_timeout(library: str, finite_horizon: bool, raw_env: Any) -> None:
    if library == "sb3":
        assert not raw_env.unwrapped.single_action_space.is_bounded("both")
    env = _wrap_env(library, raw_env)
    if library == "sb3":
        # SB3 sees normalized bounds without modifying the underlying environment.
        np.testing.assert_array_equal(env.action_space.low, -1.0)
        np.testing.assert_array_equal(env.action_space.high, 1.0)
        assert not raw_env.unwrapped.single_action_space.is_bounded("both")
    _assert_finite(env.reset())
    if library == "torchrl":
        from torchrl.envs.utils import check_env_specs

        check_env_specs(env)
        with torch.inference_mode():
            rollout = env.rollout(_EPISODE_STEPS + 1, break_when_any_done=False)
        _assert_finite(rollout)
        rewards = rollout["next", "reward"]
        dones = rollout["next", "done"]
        assert rewards.shape[:2] == dones.shape[:2] == (_NUM_ENVS, _EPISODE_STEPS + 1)
        assert torch.equal(dones, rollout["next", "terminated"] | rollout["next", "truncated"])
        assert bool(dones.any()), "The short episode must exercise automatic reset"
        assert bool(rollout["next", "truncated"].any()) is not finite_horizon
        return
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


def test_torchrl_actor_uses_unbatched_action_bounds() -> None:
    """Bounded policies must support minibatches whose size differs from the environment batch."""
    from types import SimpleNamespace

    pytest.importorskip("torchrl")
    from torchrl.data import Bounded, Composite, Unbounded

    from isaaclab_rl.torchrl import make_actor

    env = SimpleNamespace(
        batch_size=torch.Size([2]),
        action_spec=Bounded(low=-2.0, high=2.0, shape=(2, 1)),
        action_spec_unbatched=Bounded(low=-2.0, high=2.0, shape=(1,)),
        observation_spec=Composite(policy=Unbounded(shape=(2, 4)), shape=(2,)),
    )
    cfg = SimpleNamespace(actor_hidden_dims=[8], activation="ELU", init_noise_std=1.0)

    actor = make_actor(env, cfg)
    batch = TensorDict({"policy": torch.randn(6, 4)}, batch_size=[6])

    assert actor(batch)["action"].shape == (6, 1)


def _assert_observation_buffer(env: Any) -> None:
    observations = env.get_observations()
    assert isinstance(observations, TensorDict)
    assert set(observations.keys()) == set(env.unwrapped.obs_buf)
    for key, value in env.unwrapped.obs_buf.items():
        torch.testing.assert_close(observations[key], value)


def test_rsl_rl_wrapper_reports_invalid_unwrapped_type() -> None:
    """Validation errors identify the unsupported unwrapped environment."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    class UnsupportedEnv:
        pass

    class OuterEnv:
        unwrapped = UnsupportedEnv()

    with pytest.raises(ValueError, match="UnsupportedEnv") as exc_info:
        RslRlVecEnvWrapper(OuterEnv())

    assert "OuterEnv" not in str(exc_info.value)


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
