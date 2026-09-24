# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for shared and backend-specific LEAPP recurrent-state helpers."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _load_export_common_module() -> ModuleType:
    """Load shared LEAPP export helpers from the installed package."""
    return importlib.import_module("isaaclab_rl.entrypoints.backends.export_common")


def _load_backend_export_module(backend: str) -> ModuleType:
    """Load an exporter when its optional backend package is installed."""
    pytest.importorskip(backend)
    return importlib.import_module(f"isaaclab_rl.entrypoints.backends.export_{backend}")


class TestSharedRecurrentState:
    """Tests for recurrent-state helpers shared by the export backends."""

    def test_checkpoint_path_naturally_sorts_numbered_files(self, tmp_path):
        """Select the latest numbered checkpoint without importing the task package."""
        export_common = _load_export_common_module()
        checkpoint_dir = tmp_path / "run" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "model_9.pt").touch()
        expected = checkpoint_dir / "model_10.pt"
        expected.touch()

        resolved = export_common.get_checkpoint_path(
            str(tmp_path),
            run_dir="run",
            checkpoint=r"model_.*\.pt",
            other_dirs=["checkpoints"],
        )

        assert resolved == str(expected)

    def test_lstm_state_detection_requires_two_tensors(self):
        """Only two-tensor recurrent state is treated as LSTM feedback."""
        export_common = _load_export_common_module()
        hidden_state = torch.zeros(1, 1, 4)
        cell_state = torch.zeros(1, 1, 4)

        assert export_common.is_two_tensor_lstm_state([hidden_state, cell_state])
        assert export_common.is_two_tensor_lstm_state((hidden_state, cell_state))
        assert not export_common.is_two_tensor_lstm_state([hidden_state])
        assert not export_common.is_two_tensor_lstm_state([hidden_state, cell_state, cell_state])
        assert not export_common.is_two_tensor_lstm_state([hidden_state, object()])

    def test_state_sequence_round_trip_from_dict(self):
        """Named LEAPP state maps back to framework state order."""
        export_common = _load_export_common_module()
        states = [torch.zeros(1, 1, 4), torch.ones(1, 1, 4)]
        state_dict = export_common.state_dict_from_sequence(states)

        restored = export_common.state_sequence_from_registered(state_dict, list(state_dict.keys()), states)

        assert list(state_dict.keys()) == ["actor_state_0", "actor_state_1"]
        assert restored == states

    def test_graph_configs_include_controller_owned_write_contract(self):
        """Preserve controller responsibilities in graph-level LEAPP metadata."""
        pytest.importorskip("leapp")
        export_common = _load_export_common_module()
        env_cfg = types.SimpleNamespace(sim=types.SimpleNamespace(dt=0.01), decimation=2)
        requirement = {
            "capability": "gravity_compensation",
            "source_term": "arm_action",
            "kind": "target/joint/effort",
            "element_names": [["joint_a"]],
            "cadence": "action_apply",
            "isaaclab_connection": "write:robot:set_joint_effort_target_index",
        }

        graph_configs = export_common.create_graph_configs(
            env_cfg,
            controller_owned_write_requirements=(requirement,),
        )

        assert graph_configs.to_dict() == {
            "frequency": 50.0,
            "isaaclab": {
                "controller_owned_writes": {
                    "schema_version": 1,
                    "requirements": [requirement],
                }
            },
        }


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_rl_games_recurrent_state(rnn_type: Literal["lstm", "gru"]) -> None:
    """RL-Games accepts LSTM feedback and rejects GRU feedback."""
    pytest.importorskip("rl_games")
    import gymnasium as gym
    from rl_games.algos_torch.players import PpoPlayerContinuous

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    params = load_cfg_from_registry("Isaac-Cartpole", "rl_games_cfg_entry_point")["params"]
    params["network"]["rnn"] = {"name": rnn_type, "units": 7, "layers": 1}
    params["config"].update(
        device_name="cpu",
        env_info={
            "observation_space": gym.spaces.Box(-1.0, 1.0, shape=(4,)),
            "action_space": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        },
    )
    agent = PpoPlayerContinuous(params)
    agent.reset()
    export_module = _load_backend_export_module("rl_games")
    if rnn_type == "gru":
        assert not export_module.is_rl_games_lstm_policy(agent)
        with pytest.raises(NotImplementedError, match="Only RL-Games LSTM"):
            export_module._validate_rl_games_recurrent_support(agent)
    else:
        assert export_module.is_rl_games_lstm_policy(agent)
        agent.get_action(torch.zeros(4), is_deterministic=True)
        states = export_module.get_rl_games_policy_states(agent)
        assert len(states) == 2
        for actual, expected in zip(states, agent.states):
            torch.testing.assert_close(actual, expected)
            assert actual.shape == (1, 1, 7)


def _make_skrl_recurrent_agent(rnn_type: Literal["lstm", "gru"] = "lstm") -> Any:
    pytest.importorskip("skrl")
    import gymnasium as gym
    from skrl.agents.torch.ppo.ppo_rnn import PPO_RNN
    from skrl.models.torch import GaussianMixin, Model

    class _TinySkrlRecurrentPolicy(GaussianMixin, Model):
        """Recurrent Gaussian policy used by the skrl tests."""

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
            rnn_cls = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU
            self.rnn = rnn_cls(self.num_observations, 5, 1)
            self.head = torch.nn.Linear(5, self.num_actions)
            self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

        def get_specification(self):
            return {"rnn": {"sizes": [(1, 1, 5)] * (2 if rnn_type == "lstm" else 1), "sequence_length": 1}}

        def compute(self, inputs, role):
            state = tuple(inputs["rnn"]) if rnn_type == "lstm" else inputs["rnn"][0]
            out, state = self.rnn(inputs["observations"].unsqueeze(0), state)
            states = list(state) if rnn_type == "lstm" else [state]
            return self.head(out.squeeze(0)), {"log_std": self.log_std_parameter, "rnn": states}

    class _TinySkrlValue(Model):
        """Value model used by the skrl tests."""

        def __init__(self, observation_space, action_space, device):
            super().__init__(observation_space=observation_space, action_space=action_space, device=device)
            self.net = torch.nn.Linear(self.num_observations, 1)

        def compute(self, inputs, role):
            return self.net(inputs["observations"]), {}

        def act(self, inputs, role=""):
            return self.compute(inputs, role)

    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    policy = _TinySkrlRecurrentPolicy(obs_space, act_space, "cpu")
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
        """skrl LSTM feedback is detected and updated from action output."""
        export_module = _load_backend_export_module("skrl")
        agent = _make_skrl_recurrent_agent()

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
        """skrl recurrent policies without two LSTM tensors are rejected."""
        pytest.importorskip("skrl")
        export_module = _load_backend_export_module("skrl")
        agent = _make_skrl_recurrent_agent("gru")

        assert not export_module.is_skrl_lstm_policy(agent)
        with pytest.raises(NotImplementedError, match="Only skrl LSTM"):
            export_module._validate_skrl_recurrent_support(agent)


class TestRslRlRecurrentState:
    """Tests for RSL-RL recurrent-state adaptation."""

    def test_modular_rnn_model_lstm_round_trip(self):
        """LSTM state registration supports the RSL-RL 5.x RNNModel."""
        from rsl_rl.models import RNNModel
        from tensordict import TensorDict

        export_module = _load_backend_export_module("rsl_rl")
        policy = RNNModel(
            TensorDict({"policy": torch.zeros(1, 2)}, batch_size=[1]),
            {"actor": ["policy"]},
            "actor",
            output_dim=1,
            hidden_dims=[4],
            rnn_hidden_dim=4,
            rnn_num_layers=2,
        )

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
