# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SB3-specific LEAPP export helpers."""

import importlib
from pathlib import Path
from types import ModuleType

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
stable_baselines3 = pytest.importorskip("stable_baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sb3_contrib = pytest.importorskip("sb3_contrib")


def _load_export_module() -> ModuleType:
    """Load the installed SB3 exporter."""
    from isaaclab_rl.entrypoints.backends import export_sb3 as module

    return module


def test_sb3_export_args_use_common_defaults(monkeypatch):
    """Use the shared export flags and omit training-only arguments."""
    export_module = _load_export_module()
    export_common = importlib.import_module("isaaclab_rl.entrypoints.backends.export_common")

    class _AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            return None

    monkeypatch.setattr(export_common, "AppLauncher", _AppLauncher)
    monkeypatch.setattr(
        export_common,
        "setup_preset_cli",
        lambda parser, argv=None, **kwargs: parser.parse_known_args(argv),
    )
    args, _ = export_module.parse_export_args(["--task", "Isaac-Cartpole"])

    assert args.agent == "sb3_cfg_entry_point"
    assert args.headless
    assert not hasattr(args, "seed")
    assert {
        "task",
        "agent",
        "checkpoint",
        "export_task_name",
        "export_method",
        "export_save_path",
        "validation_steps",
        "disable_graph_visualization",
    }.issubset(vars(args))


def test_sb3_vec_normalize_path_matches_play_convention():
    """Derive the normalization sidecar name used by SB3 play."""
    export_module = _load_export_module()

    assert export_module._vec_normalize_path("/tmp/run/model.zip") == Path("/tmp/run/model_vecnormalize.pkl")
    assert export_module._vec_normalize_path("/tmp/run/model_100_steps.zip") == Path(
        "/tmp/run/model_vecnormalize_100_steps.pkl"
    )


def test_sb3_observation_normalization_uses_torch():
    """Apply saved VecNormalize statistics without converting to NumPy."""
    export_module = _load_export_module()
    vec_normalize = VecNormalize(DummyVecEnv([lambda: gym.make("Pendulum-v1")]), clip_obs=1.0)
    try:
        vec_normalize.obs_rms.update(np.array([[1.0, 2.0, 3.0], [3.0, 8.0, 4.0]]))
        obs = torch.tensor([[3.0, 20.0, -5.0]])
        normalized = export_module.normalize_observation(obs, vec_normalize)
        torch.testing.assert_close(normalized, torch.from_numpy(vec_normalize.normalize_obs(obs.numpy())))
    finally:
        vec_normalize.close()


def test_sb3_feedforward_actions_match_predict_clipping():
    """Compare exported inference with SB3 predict, including action clipping."""
    export_module = _load_export_module()
    agent = stable_baselines3.PPO("MlpPolicy", "Pendulum-v1", n_steps=8, batch_size=8, device="cpu")
    try:
        with torch.no_grad():
            agent.policy.action_net.weight.zero_()
            agent.policy.action_net.bias.fill_(10.0)
        obs = torch.zeros(1, 3)
        actions, state = export_module._policy_actions(agent.policy, obs)
        expected, _ = agent.predict(obs.numpy(), deterministic=True)
        assert state is None
        torch.testing.assert_close(actions, torch.from_numpy(expected))
        assert np.array_equal(expected[0], agent.action_space.high)
    finally:
        agent.env.close()


def test_sb3_recurrent_checkpoint_and_state_round_trip(tmp_path: Path) -> None:
    """A recurrent checkpoint preserves actions and actor state after loading."""
    sb3_contrib = pytest.importorskip("sb3_contrib")
    export_module = _load_export_module()
    agent = sb3_contrib.RecurrentPPO(
        "MlpLstmPolicy",
        "Pendulum-v1",
        n_steps=8,
        batch_size=8,
        policy_kwargs={"lstm_hidden_size": 4},
    )
    try:
        checkpoint_path = tmp_path / "model.zip"
        agent.save(checkpoint_path)
    finally:
        agent.env.close()

    loaded_agent = export_module._load_agent(str(checkpoint_path), device="cpu")
    assert isinstance(loaded_agent, sb3_contrib.RecurrentPPO)
    policy = loaded_agent.policy
    state = export_module.initialize_sb3_recurrent_state(policy, num_envs=1)
    observations = torch.zeros(1, 3, device=policy.device)
    actions, next_state = export_module._policy_actions(policy, observations, state)
    expected_actions, expected_state = loaded_agent.predict(observations.cpu().numpy(), deterministic=True)

    assert export_module.is_sb3_recurrent_policy(policy)
    torch.testing.assert_close(actions, torch.as_tensor(expected_actions, device=policy.device))
    assert len(next_state) == 2
    for actual, expected in zip(next_state, expected_state):
        torch.testing.assert_close(actual, torch.as_tensor(expected, device=policy.device))
