# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LEAPP checkpoint, policy, compilation, and replay checks without a physics scene."""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from leapp_initialized_checkpoints import (
    _CheckpointEnv,
    _create_rl_games_checkpoint,
    _create_rsl_rl_checkpoint,
    _create_sb3_checkpoint,
    _create_skrl_checkpoint,
)
from leapp_tensor_env import TensorManagerEnv
from leapp_test_tasks import MANAGER_TASK

from isaaclab.envs import DirectRLEnv
from isaaclab.managers import ActionManager, ObservationManager

from isaaclab_tasks.utils import load_cfg_from_registry

_UR10_TASK = "Isaac-Reach-UR10"
_DIRECT_TASK = "Isaac-Leapp-Direct-Tensor"
_CREATORS = {
    "rsl_rl": _create_rsl_rl_checkpoint,
    "rl_games": _create_rl_games_checkpoint,
    "skrl": _create_skrl_checkpoint,
    "sb3": _create_sb3_checkpoint,
}


class _TensorEnv:
    """Provide framework observations and LEAPP I/O edges without a simulation context."""

    def __init__(self, cfg, *, task: str, checkpoint_task: str, backend: str, render_mode=None):
        self.cfg = cfg
        self._task = task
        self.render_mode = render_mode
        self.spec = SimpleNamespace(id=task)
        self._source = _CheckpointEnv(checkpoint_task, backend)
        self._recording = False
        self.action_manager = SimpleNamespace(total_action_dim=self._source.num_actions)
        self.single_action_space = self._source.action_space
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, 1)
        source_spaces = {
            name: gym.spaces.Box(-np.inf, np.inf, tuple(value.shape[1:]), dtype=np.float32)
            for name, value in self._source.get_observations().items()
        }
        if backend == "sb3":
            source_spaces["policy"] = self._source.observation_space
        self.single_observation_space = gym.spaces.Dict(source_spaces)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, 1)
        self.obs_buf = self._observations()

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def max_episode_length(self) -> int:
        return 100

    def seed(self, seed: int = -1) -> int:
        return seed

    def _observations(self) -> dict:
        if isinstance(self._source.observation_space, gym.spaces.Dict):
            policy = {
                name: torch.full((1, *space.shape), 0.25, dtype=torch.float32)
                for name, space in self._source.observation_space.spaces.items()
            }
            return {"policy": policy, "critic": torch.zeros(1, 2)}
        return self._source.get_observations().to_dict()

    def _annotate_observations(self, observations: dict) -> dict:
        from leapp import annotate

        flat = {
            f"{group}_{term}" if isinstance(value, dict) else group: tensor
            for group, value in observations.items()
            for term, tensor in (value.items() if isinstance(value, dict) else [(None, value)])
        }
        values = annotate.input_tensors(self._task, flat)
        annotated = dict(zip(flat, (values,) if len(flat) == 1 else values))
        return {
            group: {term: annotated[f"{group}_{term}"] for term in value}
            if isinstance(value, dict)
            else annotated[group]
            for group, value in observations.items()
        }

    def reset(self, *, seed=None, options=None):
        self.obs_buf = self._observations()
        if self._recording:
            self.obs_buf = self._annotate_observations(self.obs_buf)
        return self.obs_buf, {}

    def step(self, actions):
        if self._recording:
            from leapp import annotate

            annotate.output_tensors(self._task, {"actions": actions}, export_with="onnx-dynamo")
            self.obs_buf = self._annotate_observations(self._observations())
        return (
            self.obs_buf,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    def close(self):
        """The tensor fixture owns no simulator resources."""
        pass


class _DirectTensorEnv(_TensorEnv, DirectRLEnv):
    pass


@pytest.mark.parametrize(
    ("backend", "task", "checkpoint_task", "direct"),
    [
        pytest.param("rsl_rl", MANAGER_TASK, MANAGER_TASK, False, id="rsl-rl-cnn"),
        pytest.param("rl_games", MANAGER_TASK, MANAGER_TASK, False, id="rl-games-lstm"),
        pytest.param("skrl", MANAGER_TASK, MANAGER_TASK, False, id="skrl"),
        pytest.param("sb3", MANAGER_TASK, MANAGER_TASK, False, id="sb3-dict-normalization"),
        pytest.param("rsl_rl", _DIRECT_TASK, MANAGER_TASK, True, id="direct-rsl-rl"),
        pytest.param("rsl_rl", _UR10_TASK, _UR10_TASK, False, id="ur10-rsl-rl"),
    ],
)
def test_leapp_export_flow(backend: str, task: str, checkpoint_task: str, direct: bool, tmp_path: Path, monkeypatch):
    """Load a native checkpoint, export the real backend policy, and replay changed observations."""
    from leapp import InferenceManager

    from isaaclab_rl.entrypoints.backends import export_rl_games, export_rsl_rl, export_sb3, export_skrl

    modules = {"rsl_rl": export_rsl_rl, "rl_games": export_rl_games, "skrl": export_skrl, "sb3": export_sb3}
    module = modules[backend]
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    agent_cfg = load_cfg_from_registry(checkpoint_task, f"{backend}_cfg_entry_point")
    checkpoint = _CREATORS[backend](_CheckpointEnv(checkpoint_task, backend), agent_cfg, tmp_path)
    env_cfg = load_cfg_from_registry(checkpoint_task, "env_cfg_entry_point")
    connected = backend == "rsl_rl" and task == MANAGER_TASK and not direct
    env = (
        _DirectTensorEnv(env_cfg, task=task, checkpoint_task=checkpoint_task, backend=backend)
        if direct
        else TensorManagerEnv(
            env_cfg, task=task, backend=backend, width=_CheckpointEnv(checkpoint_task, backend).num_actions
        )
    )
    if not direct:
        assert isinstance(env.observation_manager, ObservationManager)
        assert isinstance(env.action_manager, ActionManager)
    if connected:
        observed = env.observation_manager.compute()["policy"]
        positions = observed[:, : env.action_manager.total_action_dim]
        torch.testing.assert_close(positions, torch.full_like(positions, 0.7))
        env.action_manager.process_action(torch.full((1, env.action_manager.total_action_dim), 0.4))
        env.action_manager.apply_action()
        torch.testing.assert_close(env.scene["robot"].targets[-1]["value"], torch.full_like(positions, 0.3))
        env.action_manager.process_action(torch.full((1, env.action_manager.total_action_dim), 5.0))
        env.action_manager.apply_action()
        torch.testing.assert_close(env.scene["robot"].targets[-1]["value"], torch.full_like(positions, 0.5))
        env.action_manager.reset()
    monkeypatch.setattr(module.gym, "make", lambda *args, **kwargs: env)
    real_capture = module.leapp_capture

    @contextlib.contextmanager
    def capture(*args, **kwargs):
        with real_capture(*args, **kwargs) as steps:
            env._recording = True
            try:
                yield steps
            finally:
                env._recording = False

    monkeypatch.setattr(module, "leapp_capture", capture)
    args, _ = module.parse_export_args(
        [
            "--task",
            task,
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--export_save_path",
            str(tmp_path),
            "--validation_steps",
            "2",
            "--disable_graph_visualization",
        ]
    )
    assert getattr(module, f"export_{backend}_agent")(args, env_cfg, agent_cfg)

    artifact_dir = tmp_path / task
    pipeline = artifact_dir / f"{task}.yaml"
    assert pipeline.is_file()
    assert (artifact_dir / f"{task}.onnx").is_file()
    description = yaml.safe_load(pipeline.read_text())
    manager = InferenceManager(str(pipeline))
    if not direct:
        model = description["models"][task]
        connections = {item["isaaclab_connection"] for item in model["inputs"] if "isaaclab_connection" in item}
        assert {"state:robot:joint_pos", "state:robot:joint_vel"} <= connections
        action_output = next(item for item in model["outputs"] if item["name"] == "actions")
        assert action_output["isaaclab_connection"] == "write:robot:set_joint_position_target_index"
        if backend != "sb3":
            assert f"{task}/h_policy_joint_vel_out" in description["pipeline"]["feedback_flow"]
    if connected:
        from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv

        model = description["models"][task]
        assert {item["isaaclab_connection"] for item in model["inputs"] if "isaaclab_connection" in item} == {
            "state:camera:output.rgb",
            "state:robot:joint_pos",
            "state:robot:joint_vel",
        }
        assert [
            (item["name"], item["isaaclab_connection"]) for item in model["outputs"] if "isaaclab_connection" in item
        ] == [("actions", "write:robot:set_joint_position_target_index")]
        action_output = next(item for item in model["outputs"] if item["name"] == "actions")
        assert action_output["element_names"] == [[f"joint_{index}" for index in range(10)]]
        deployment = object.__new__(LeappDeploymentEnv)
        deployment.scene = env.scene
        deployment._leapp_desc = description
        deployment.inference = manager
        deployment._input_mapping = {}
        deployment._output_mapping = {}
        deployment.command_manager = None
        deployment._resolve_io()
        manager.reset()
        deployed_outputs = manager.run_policy(deployment._read_inputs())
        deployment._write_outputs(deployed_outputs)
        torch.testing.assert_close(env.scene["robot"].targets[-1]["value"], deployed_outputs[f"{task}/actions"])
    inputs = {}
    for node_name, names in description["pipeline"]["inputs"].items():
        for desc in description["models"][node_name]["inputs"]:
            if desc["name"] in names:
                inputs[f"{node_name}/{desc['name']}"] = torch.full(
                    desc["shape"], 0.25, dtype=getattr(torch, desc["dtype"])
                )
    assert inputs
    manager.reset()
    baseline = {key: value.clone() for key, value in manager.run_policy(inputs).items()}
    action_keys = baseline.keys() - description["pipeline"]["feedback_flow"].keys()
    assert action_keys
    for name, value in inputs.items():
        manager.reset()
        changed = {**inputs, name: value + (127 if value.dtype == torch.uint8 else 0.5)}
        replay = manager.run_policy(changed)
        assert all(torch.isfinite(output).all() for output in replay.values())
        assert any(not torch.allclose(replay[key], baseline[key], atol=1e-7) for key in action_keys), name
    sequences = []
    for _ in range(2):
        manager.reset()
        sequences.append([{key: value.clone() for key, value in manager.run_policy(inputs).items()} for _ in range(2)])
    for first, second in zip(*sequences):
        for name in first:
            torch.testing.assert_close(first[name], second[name])
    if backend == "rsl_rl" and not direct and task == MANAGER_TASK:
        import onnx

        assert any(node.op_type == "Conv" for node in onnx.load(artifact_dir / f"{task}.onnx").graph.node)
    if backend == "rl_games":
        assert len(description["pipeline"]["feedback_flow"]) >= 2


def test_fast_export_has_no_simulator_dependency():
    """Keep the export flow independent of scene and solver setup."""
    export_dir = Path(__file__).parent
    for obsolete_file in ("test_leapp_export_simulation.py", "leapp_synthetic_env.py", "leapp_synthetic_env_cfg.py"):
        assert not (export_dir / obsolete_file).exists()
    forbidden_modules = {"isaaclab.sim", "isaaclab_newton", "isaaclab_physx"}
    forbidden_calls = {"launch_simulation", "SimulationContext"}
    for filename in ("test_leapp_export_flow.py", "leapp_tensor_env.py"):
        for node in ast.walk(ast.parse((export_dir / filename).read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not any(module == name or module.startswith(name + ".") for name in forbidden_modules)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setattr":
                assert not any(
                    isinstance(arg, ast.Constant) and arg.value in {"prepare_export_env", "BaseArticulation"}
                    for arg in node.args
                )
