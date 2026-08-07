# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SB3-specific LEAPP export helpers."""

import importlib.util
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
stable_baselines3 = pytest.importorskip("stable_baselines3")
sb3_contrib = pytest.importorskip("sb3_contrib")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "sb3" / "export.py"
_EXPORT_MODULE_NAME = "_isaaclab_sb3_leapp_export"


def _load_export_module():
    """Load SB3 export.py without importing Isaac Sim runtime modules."""
    sys.modules.pop(_EXPORT_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_EXPORT_MODULE_NAME, _EXPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXPORT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    module.torch = torch
    return module


def test_sb3_export_args_use_common_defaults():
    """Use the shared export flags and omit training-only arguments."""
    export_module = _load_export_module()

    original_modules = {
        name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.app", "isaaclab_tasks", "isaaclab_tasks.utils")
    }
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_app_module = types.ModuleType("isaaclab.app")
    isaaclab_tasks_module = types.ModuleType("isaaclab_tasks")
    isaaclab_tasks_utils_module = types.ModuleType("isaaclab_tasks.utils")

    class _AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            return None

    setattr(isaaclab_app_module, "AppLauncher", _AppLauncher)
    setattr(
        isaaclab_tasks_utils_module,
        "setup_preset_cli",
        lambda parser, argv=None, **kwargs: parser.parse_known_args(argv),
    )
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = isaaclab_app_module
    sys.modules["isaaclab_tasks"] = isaaclab_tasks_module
    sys.modules["isaaclab_tasks.utils"] = isaaclab_tasks_utils_module
    try:
        args, _ = export_module.parse_export_args(["--task", "Isaac-Cartpole"])
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module

    assert args.agent == "sb3_cfg_entry_point"
    assert args.headless
    assert not hasattr(args, "seed")
    assert {
        "task",
        "agent",
        "checkpoint",
        "use_pretrained_checkpoint",
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
    running_stats = types.SimpleNamespace(mean=np.array([1.0, 2.0]), var=np.array([4.0, 9.0]))
    vec_normalize = types.SimpleNamespace(
        norm_obs=True,
        norm_obs_keys=None,
        obs_rms=running_stats,
        epsilon=0.0,
        clip_obs=1.0,
    )

    normalized = export_module.normalize_observation(torch.tensor([[3.0, 8.0]]), vec_normalize)

    assert torch.allclose(normalized, torch.tensor([[1.0, 1.0]]))


def test_sb3_feedforward_actions_match_predict_clipping():
    """Clip feed-forward policy actions using the saved SB3 action space."""
    export_module = _load_export_module()

    class _Policy:
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        squash_output = False

        def set_training_mode(self, mode):
            return None

        def __call__(self, obs, deterministic):
            return torch.tensor([[2.0, -2.0]]), None, None

    actions, state = export_module._policy_actions(_Policy(), torch.zeros(1, 3))

    assert state is None
    assert torch.equal(actions, torch.tensor([[1.0, -1.0]]))


def test_sb3_recurrent_policy_state_round_trip():
    """Run RecurrentPPO inference with traceable actor LSTM state tensors."""
    export_module = _load_export_module()
    agent = sb3_contrib.RecurrentPPO(
        "MlpLstmPolicy",
        "Pendulum-v1",
        n_steps=8,
        batch_size=8,
        policy_kwargs={"lstm_hidden_size": 4},
    )
    policy = agent.policy
    state = export_module.initialize_sb3_recurrent_state(policy, num_envs=1)

    actions, next_state = export_module._policy_actions(policy, torch.zeros(1, 3, device=policy.device), state)

    assert export_module.is_sb3_recurrent_policy(policy)
    assert tuple(actions.shape) == (1, 1)
    assert len(next_state) == 2
    assert all(tuple(tensor.shape) == (1, 1, 4) for tensor in next_state)
    agent.env.close()


def test_sb3_checkpoint_loader_selects_recurrent_ppo(tmp_path):
    """Select RecurrentPPO from the policy class serialized in the checkpoint."""
    export_module = _load_export_module()
    from stable_baselines3.common.save_util import load_from_zip_file

    agent = sb3_contrib.RecurrentPPO(
        "MlpLstmPolicy",
        "Pendulum-v1",
        n_steps=8,
        batch_size=8,
        policy_kwargs={"lstm_hidden_size": 4},
    )
    checkpoint_path = tmp_path / "model.zip"
    agent.save(checkpoint_path)
    agent.env.close()

    export_module.PPO = stable_baselines3.PPO
    export_module.RecurrentPPO = sb3_contrib.RecurrentPPO
    export_module.load_from_zip_file = load_from_zip_file
    loaded_agent = export_module._load_agent(str(checkpoint_path), device="cpu")

    assert isinstance(loaded_agent, sb3_contrib.RecurrentPPO)
