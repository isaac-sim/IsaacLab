# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for shared and backend-specific LEAPP recurrent-state helpers."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_LEAPP_ROOT = Path(__file__).resolve().parents[4] / "scripts" / "reinforcement_learning" / "leapp"
_EXPORT_UTILS_SCRIPT = _LEAPP_ROOT / "export_utils.py"
_EXPORT_UTILS_MODULE_NAME = "_isaaclab_leapp_export_utils"


@contextlib.contextmanager
def _stub_isaaclab_cli_imports(*, fold_preset_tokens: bool = False) -> Iterator[None]:
    """Stub Isaac Lab CLI imports so export scripts can be loaded without Kit."""
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
    if fold_preset_tokens:
        setattr(isaaclab_tasks_utils_module, "fold_preset_tokens", lambda args: args)
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
        yield
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def _load_export_utils_module() -> ModuleType:
    """Load shared LEAPP export helpers from the scripts tree."""
    sys.modules.pop(_EXPORT_UTILS_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_EXPORT_UTILS_MODULE_NAME, _EXPORT_UTILS_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {_EXPORT_UTILS_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXPORT_UTILS_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _load_backend_export_module(backend: str) -> ModuleType:
    """Load a backend export script without importing Isaac Sim runtime modules."""
    export_script = _LEAPP_ROOT / backend / "export.py"
    module_name = f"_isaaclab_{backend}_leapp_export"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, export_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {export_script}")

    module = importlib.util.module_from_spec(spec)
    with _stub_isaaclab_cli_imports(fold_preset_tokens=backend == "rsl_rl"):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    if backend in ("rl_games", "skrl"):
        setattr(module, "is_two_tensor_lstm_state", _load_export_utils_module().is_two_tensor_lstm_state)
    elif backend == "rsl_rl":
        setattr(module, "torch", torch)
    return module


class TestSharedRecurrentState:
    """Tests for recurrent-state helpers shared by the export backends."""

    def test_lstm_state_detection_requires_two_tensors(self):
        """Check that only two-tensor recurrent state is treated as LSTM feedback."""
        export_utils = _load_export_utils_module()
        h = torch.zeros(1, 1, 4)
        c = torch.zeros(1, 1, 4)

        assert export_utils.is_two_tensor_lstm_state([h, c])
        assert export_utils.is_two_tensor_lstm_state((h, c))
        assert not export_utils.is_two_tensor_lstm_state([h])
        assert not export_utils.is_two_tensor_lstm_state([h, c, c])
        assert not export_utils.is_two_tensor_lstm_state([h, object()])

    def test_state_sequence_round_trip_from_dict(self):
        """Check named LEAPP state maps back to framework state order."""
        export_utils = _load_export_utils_module()
        states = [torch.zeros(1, 1, 4), torch.ones(1, 1, 4)]
        state_dict = export_utils.state_dict_from_sequence(states)

        restored = export_utils.state_sequence_from_registered(state_dict, list(state_dict.keys()), states)

        assert list(state_dict.keys()) == ["actor_state_0", "actor_state_1"]
        assert restored == states


class TestRlGamesRecurrentState:
    """Tests for RL-Games recurrent-state adaptation."""

    def test_lstm_feedback_detection(self):
        """Verify RL-Games LSTM feedback can be detected from player state."""
        export_module = _load_backend_export_module("rl_games")
        agent = types.SimpleNamespace(
            is_rnn=True,
            states=[torch.zeros(1, 1, 7), torch.zeros(1, 1, 7)],
        )

        assert export_module.is_rl_games_lstm_policy(agent)
        assert [tuple(tensor.shape) for tensor in export_module.get_rl_games_policy_states(agent)] == [
            (1, 1, 7),
            (1, 1, 7),
        ]

    def test_recurrent_non_lstm_is_rejected(self):
        """Verify recurrent RL-Games policies without two LSTM tensors are rejected."""
        export_module = _load_backend_export_module("rl_games")
        agent = types.SimpleNamespace(is_rnn=True, states=[torch.zeros(1, 1, 7)])

        assert not export_module.is_rl_games_lstm_policy(agent)
        with pytest.raises(NotImplementedError, match="Only RL-Games LSTM"):
            export_module._validate_rl_games_recurrent_support(agent)


def _make_skrl_lstm_agent():
    pytest.importorskip("skrl")
    import gymnasium as gym
    from skrl.agents.torch.ppo.ppo_rnn import PPO_RNN
    from skrl.models.torch import GaussianMixin, Model

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

    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
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


class TestSkrlRecurrentState:
    """Tests for skrl recurrent-state adaptation."""

    def test_lstm_feedback_detection_and_output_state(self):
        """Verify skrl LSTM feedback can be detected and updated from action output."""
        export_module = _load_backend_export_module("skrl")
        agent = _make_skrl_lstm_agent()

        assert export_module.is_skrl_lstm_policy(agent)
        assert [tuple(tensor.shape) for tensor in export_module.get_skrl_policy_states(agent)] == [
            (1, 1, 5),
            (1, 1, 5),
        ]

        actions, outputs = agent.act(torch.zeros(1, 3), None, timestep=0, timesteps=1)
        output_states = export_module.get_skrl_policy_output_states(agent, outputs)

        assert tuple(actions.shape) == (1, 2)
        assert [tuple(tensor.shape) for tensor in output_states] == [(1, 1, 5), (1, 1, 5)]

    def test_recurrent_non_lstm_is_rejected(self):
        """Verify recurrent skrl policies without two LSTM tensors are rejected."""
        pytest.importorskip("skrl")
        export_module = _load_backend_export_module("skrl")
        agent = types.SimpleNamespace(
            _rnn=True,
            _rnn_initial_states={"policy": [torch.zeros(1, 1, 5)]},
            policy=types.SimpleNamespace(get_specification=lambda: {"rnn": {"sizes": [(1, 1, 5)]}}),
        )

        assert not export_module.is_skrl_lstm_policy(agent)
        with pytest.raises(NotImplementedError, match="Only skrl LSTM"):
            export_module._validate_skrl_recurrent_support(agent)


class TestRslRlRecurrentState:
    """Tests for RSL-RL recurrent-state adaptation."""

    def test_modular_rnn_model_lstm_round_trip(self):
        """Verify LSTM state registration helpers support RSL-RL 5.x RNNModel."""

        class _Memory(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=2)
                self.hidden_state = None

        class _Policy(torch.nn.Module):
            is_recurrent = True

            def __init__(self):
                super().__init__()
                self.rnn = _Memory()

            def get_hidden_state(self):
                return self.rnn.hidden_state

        export_module = _load_backend_export_module("rsl_rl")
        policy = _Policy()

        actor_state = export_module.ensure_actor_hidden_state_initialized(
            policy, batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        registered_state = tuple(tensor + 1.0 for tensor in actor_state)
        export_module.set_actor_hidden_state(
            policy,
            export_module.actor_hidden_from_registered(registered_state, actor_state),
        )

        assert export_module.is_actor_recurrent_policy(policy)
        assert export_module.get_actor_memory_module(policy) is policy.rnn
        assert export_module.get_actor_hidden_state(policy) is registered_state
        assert policy.rnn.hidden_state is registered_state
