# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import gymnasium as gym
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skrl")

from skrl.agents.torch.ppo.ppo_rnn import PPO_RNN
from skrl.models.torch import GaussianMixin, Model

from isaaclab_rl.leapp import is_two_tensor_lstm_state

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "skrl" / "export.py"
_EXPORT_MODULE_NAME = "_isaaclab_skrl_leapp_export_helpers"


def _load_export_module():
    """Load skrl export.py without importing Isaac Sim runtime modules."""
    sys.modules.pop(_EXPORT_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_EXPORT_MODULE_NAME, _EXPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)

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
    setattr(isaaclab_tasks_utils_module, "setup_preset_cli", lambda parser, argv=None: parser.parse_known_args(argv))
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = isaaclab_app_module
    sys.modules["isaaclab_tasks"] = isaaclab_tasks_module
    sys.modules["isaaclab_tasks.utils"] = isaaclab_tasks_utils_module
    try:
        sys.modules[_EXPORT_MODULE_NAME] = module
        spec.loader.exec_module(module)
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module

    module.is_two_tensor_lstm_state = is_two_tensor_lstm_state
    return module


class _TinySkrlLstmPolicy(GaussianMixin, Model):
    """Minimal skrl Gaussian policy with LSTM state specification."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            reduction="sum",
            role="policy",
        )
        self.lstm = torch.nn.LSTM(self.num_observations, 5, 1)
        self.head = torch.nn.Linear(5, self.num_actions)
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return {"rnn": {"sizes": [(1, 1, 5), (1, 1, 5)], "sequence_length": 1}}

    def compute(self, inputs, role):
        out, rnn = self.lstm(inputs["observations"].unsqueeze(0), tuple(inputs["rnn"]))
        return self.head(out.squeeze(0)), {"log_std": self.log_std_parameter, "rnn": list(rnn)}


class _TinySkrlValue(Model):
    """Minimal skrl value model."""

    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)
        self.net = torch.nn.Linear(self.num_observations, 1)

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}

    def act(self, inputs, role=""):
        return self.compute(inputs, role)


def _make_lstm_agent():
    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=float)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
    policy = _TinySkrlLstmPolicy(obs_space, act_space, "cpu")
    value = _TinySkrlValue(obs_space, act_space, "cpu")
    agent = PPO_RNN(
        models={"policy": policy, "value": value},
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        cfg={
            "experiment": {
                "write_interval": 0,
                "checkpoint_interval": 0,
                "directory": "",
                "experiment_name": "",
            }
        },
    )
    agent.init()
    return agent


def test_skrl_lstm_feedback_detection_and_output_state():
    """Verify skrl LSTM feedback can be detected and updated from action output."""
    export_module = _load_export_module()
    agent = _make_lstm_agent()

    assert export_module.is_skrl_lstm_policy(agent)
    assert [tuple(tensor.shape) for tensor in export_module.get_skrl_policy_states(agent)] == [(1, 1, 5), (1, 1, 5)]

    actions, outputs = agent.act(torch.zeros(1, 3), None, timestep=0, timesteps=1)
    output_states = export_module.get_skrl_policy_output_states(agent, outputs)

    assert tuple(actions.shape) == (1, 2)
    assert [tuple(tensor.shape) for tensor in output_states] == [(1, 1, 5), (1, 1, 5)]


def test_skrl_recurrent_non_lstm_is_rejected():
    """Verify recurrent skrl policies without two LSTM tensors are rejected."""
    export_module = _load_export_module()
    agent = types.SimpleNamespace(
        _rnn=True,
        _rnn_initial_states={"policy": [torch.zeros(1, 1, 5)]},
        policy=types.SimpleNamespace(get_specification=lambda: {"rnn": {"sizes": [(1, 1, 5)]}}),
    )

    assert not export_module.is_skrl_lstm_policy(agent)
    with pytest.raises(NotImplementedError, match="Only skrl LSTM"):
        export_module._validate_skrl_recurrent_support(agent)


def test_skrl_non_torch_framework_is_rejected(monkeypatch):
    """Verify LEAPP export clearly rejects non-torch skrl frameworks."""
    export_module = _load_export_module()
    monkeypatch.setattr(export_module, "_load_runtime_dependencies", lambda: None)
    args_cli = argparse.Namespace(ml_framework="jax")

    with pytest.raises(NotImplementedError, match="LEAPP export only supports torch"):
        export_module.export_skrl_agent(args_cli, env_cfg=None, experiment_cfg=None)
