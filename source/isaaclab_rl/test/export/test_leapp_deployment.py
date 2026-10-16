# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.assets.articulation import BaseArticulation
from isaaclab.envs import leapp_deployment_env
from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv
from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.proxy import _ArticulationWriteProxy, _DataProxy


def test_deployment_env_reuses_and_delegates_manager_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test deployment reuses its registered environment and delegates task state."""
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text("pipeline:\n  inputs: {}\n  outputs: {}\n")
    closed = []
    base_env = SimpleNamespace(
        cfg=SimpleNamespace(),
        extras={},
        sim=SimpleNamespace(),
        scene={},
        event_manager=SimpleNamespace(),
        command_manager=SimpleNamespace(),
        has_rtx_sensors=False,
        num_envs=1,
        reset_buf=torch.zeros(1, dtype=torch.bool),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(leapp_deployment_env, "InferenceManager", lambda _: SimpleNamespace(nodes={}))

    env = LeappDeploymentEnv(base_env, str(pipeline_path))

    assert env.scene is base_env.scene
    assert env.command_manager is base_env.command_manager
    assert env.reset_buf is base_env.reset_buf
    env.close()
    env.close()
    assert closed == [True]


def test_fixed_tendon_output_mapping_round_trips_through_deployment():
    """Test fixed-tendon targets retain their names and deployment selector."""

    class _TendonWriteSemantics:
        set_fixed_tendon_position_target_index = BaseArticulation.set_fixed_tendon_position_target_index

    class _TendonArticulation(_TendonWriteSemantics):
        joint_names = ["joint_0", "joint_1"]
        fixed_tendon_names = ["tendon_0", "tendon_1", "tendon_2"]
        data = SimpleNamespace()

        def __init__(self):
            self.writes = []

        def find_fixed_tendons(self, names, preserve_order=False):
            assert preserve_order
            return [self.fixed_tendon_names.index(name) for name in names], names

        def set_fixed_tendon_position_target_index(self, *, target, fixed_tendon_ids=None, env_ids=None):
            self.writes.append((target.clone(), fixed_tendon_ids, env_ids))

    asset = _TendonArticulation()
    outputs = []
    proxy = _ArticulationWriteProxy(
        real_asset=asset,
        entity_name="robot",
        term_name="tendon_pos",
        output_cache=outputs,
        method_resolution_cache={},
        captured_write_term_names=set(),
        data_proxy=_DataProxy(asset.data, "robot", "tendon-task", {}, {}, lambda name: name),
    )
    target = torch.tensor([[0.2, 0.4]])
    tendon_ids = torch.tensor([2, 0], dtype=torch.int32)

    proxy.set_fixed_tendon_position_target_index(target=target, fixed_tendon_ids=tendon_ids)

    assert len(outputs) == 1
    semantics = outputs[0]
    assert semantics.name == "tendon_pos"
    assert semantics.kind == "target/tendon/position"
    assert semantics.element_names == [["tendon_2", "tendon_0"]]
    assert semantics.extra == {"isaaclab_connection": "write:robot:set_fixed_tendon_position_target_index"}

    asset.writes.clear()
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"robot": asset}
    env.inference = SimpleNamespace(
        nodes={
            "tendon-task": SimpleNamespace(
                input_descriptions=[],
                output_descriptions=[{"name": semantics.name, **semantics.to_dict()}],
            )
        }
    )
    env._leapp_desc = {"pipeline": {"inputs": {}, "outputs": {"tendon-task": [semantics.name]}}}
    env._input_mapping = {}
    env._output_mapping = {}
    env._resolve_io()
    env._write_outputs({"tendon-task/tendon_pos": target})

    assert len(asset.writes) == 1
    written_target, written_ids, written_env_ids = asset.writes[0]
    assert torch.equal(written_target, target)
    assert written_ids == [2, 0]
    assert written_env_ids is None

    tendon_term = SimpleNamespace(_asset=asset, _tendon_ids=tendon_ids)
    assert (
        ExportPatcher("onnx-dynamo")._collect_action_static_outputs(SimpleNamespace(_terms={"tendon_pos": tendon_term}))
        == []
    )
