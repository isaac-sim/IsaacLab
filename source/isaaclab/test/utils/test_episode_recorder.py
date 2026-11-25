# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import ast
import inspect
import sys
from types import SimpleNamespace

import h5py
import pytest
import torch
import warp as wp

from isaaclab.envs.mdp.recorders import record_pre_step
from isaaclab.utils.datasets import DatasetExportMode, EpisodeData, EpisodeRecorder, EpisodeRecorderCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_batched_episode_indices_and_ownership(monkeypatch):
    """Asynchronous resets, field cadence, growth and snapshots must preserve exact samples."""
    cfg = EpisodeRecorderCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE, buffer_initial_capacity=2)
    recorder = EpisodeRecorder(cfg, num_envs=3, device="cpu")
    recorder.add_to_episodes("unused", torch.empty(0, 7), [])
    assert not recorder._fields
    with pytest.raises(ValueError, match="requires 3 rows"):
        recorder.add_to_episodes("invalid", torch.empty(2, 7))
    assert not recorder._fields
    source = torch.tensor([[1.0, -0.0], [2.0, float("nan")], [3.0, 4.0]])
    # Policies commonly collect under inference_mode, then reset/export after leaving it.
    with torch.inference_mode():
        recorder.add_to_episodes("obs", {"state": source})
    wp.init()
    recorder.add_to_episodes("initial_state", wp.from_torch(torch.tensor([[10], [20], [30]], dtype=torch.int32)))
    source.fill_(99)
    recorder.add_to_episodes("obs/state", source)
    snapshot = recorder.get_episode(1)
    assert snapshot.data["obs"]["state"][0, 0] == 2
    assert torch.isnan(snapshot.data["obs"]["state"][0, 1])
    assert torch.signbit(recorder.get_episode(0).data["obs"]["state"][0, 1])
    assert snapshot.data["initial_state"].dtype == torch.int32
    assert snapshot.success is None

    recorder.reset(torch.tensor([1]))
    with torch.inference_mode():
        recorder.add_to_episodes("obs/state", torch.tensor([[42.0, 43.0], [52.0, 53.0]]), [2, 1])
    recorder.add_to_episodes("obs/state", torch.tensor([[62.0, 63.0]]), slice(0, 1))
    assert torch.equal(recorder.get_episode(1).data["obs"]["state"], torch.tensor([[52.0, 53.0]]))
    assert recorder.get_episode(1).data.keys() == {"obs"}
    assert torch.equal(recorder.get_episode(2).data["obs"]["state"][:, 0], torch.tensor([3.0, 99.0, 42.0]))
    assert torch.equal(recorder.get_episode(0).data["obs"]["state"][:, 0], torch.tensor([1.0, 99.0, 62.0]))
    assert snapshot.data["obs"]["state"].shape == (2, 2)
    snapshot = recorder.get_episode(1)
    snapshot.data["obs"]["state"].fill_(-100)
    assert recorder.get_episode(1).data["obs"]["state"][0, 0] == 52

    with pytest.raises(ValueError, match="changed shape"):
        recorder.add_to_episodes("obs/state", torch.zeros(3, 5))
    with pytest.raises(ValueError, match="conflicts"):
        recorder.add_to_episodes("obs", torch.zeros(3, 2))
    recorder.add_to_episodes("obs/state", torch.empty(0, 2), [])
    assert recorder.get_episode(0).data["obs"]["state"].shape == (3, 2)

    recorder.reset()
    # A new schema is allowed once every world has released this field, e.g. manual annotation.
    with torch.inference_mode():
        recorder.add_to_episodes("obs/state", torch.zeros(3, 5, dtype=torch.bool))
    assert recorder.get_episode(0).data["obs"]["state"].dtype == torch.bool
    recorder.reset()

    def forbidden(*args, **kwargs):
        raise AssertionError("Steady-state recording must reuse storage without episode objects or host index reads")

    # Architectural/performance gate: preallocated insertion and reset need neither episode objects,
    # per-world container appends, nor device-to-host scalar reads. Export-none also stays batched.
    with monkeypatch.context() as patch:
        patch.setattr(EpisodeData, "__init__", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch, "empty", forbidden)
        recorder.add_to_episodes("obs/state", torch.zeros(3, 5, dtype=torch.bool))
        recorder.set_success_to_episodes([0, 2], torch.tensor([[True], [False]]))
        recorder.export_episodes([0, 2])
        recorder.reset([1])
    assert recorder.exported_successful_episode_count == 1
    assert recorder.exported_failed_episode_count == 1
    assert all(recorder.get_episode(i).is_empty() for i in range(3))
    # Capture requested actions in the action manager's applied dtype, before caller buffers change.
    requested = torch.tensor([[1.25, 2.5], [3.75, 4.0], [5.5, 6.25]], dtype=torch.float64)
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.zeros(3, 2)),
        obs_buf={"policy": {"state": torch.ones(3, 5, dtype=torch.bool)}},
    )
    record_pre_step(recorder, env, requested)
    requested.zero_()
    env.obs_buf["policy"]["state"].zero_()
    captured = recorder.get_episode(1)
    torch.testing.assert_close(captured.data["actions"], torch.tensor([[3.75, 4.0]]))
    assert captured.data["obs"]["state"].all()
    recorder.close()


@pytest.mark.parametrize("mode", list(DatasetExportMode))
def test_export_batched_episodes(tmp_path, mode):
    """Export modes, custom IDs, nested tensor schema and counters survive partial finalization."""
    cfg = EpisodeRecorderCfg(dataset_export_dir_path=str(tmp_path), dataset_export_mode=mode, export_in_close=True)
    recorder = EpisodeRecorder(cfg, num_envs=3, device="cpu")
    values = torch.tensor([[1.0, -0.0], [2.0, float("nan")], [3.0, 4.0]])
    recorder.add_to_episodes("actions", values)
    recorder.add_to_episodes("obs", {"flag": torch.tensor([True, False, True])})
    recorder.set_success_to_episodes(None, torch.tensor([[True], [False], [True]]))
    with pytest.raises(ValueError, match="must match"):
        recorder.export_episodes([2, 1], [30])
    with pytest.raises(ValueError, match="must be unique"):
        recorder.export_episodes([2, 1], [30, 30])
    recorder.export_episodes(torch.tensor([2, 1]), [30, 10])
    assert recorder.get_episode(1).is_empty()
    assert recorder.get_episode(2).is_empty()
    assert recorder.get_episode(0).success is True
    recorder.export_episodes(slice(0, 1), [20])
    recorder.export_episodes()  # Empty episodes do not affect counters.
    assert recorder.exported_successful_episode_count == 2
    assert recorder.exported_failed_episode_count == 1
    recorder.close()
    if mode == DatasetExportMode.EXPORT_NONE:
        assert list(tmp_path.iterdir()) == []
        return
    with h5py.File(tmp_path / "dataset.hdf5") as file:
        expected = {"demo_30", "demo_20"}
        if mode == DatasetExportMode.EXPORT_ALL:
            expected.add("demo_10")
        assert set(file["data"]) == expected
        assert file["data/demo_20/actions"][()].tobytes() == values[0:1].numpy().tobytes()
        assert bool(file["data/demo_30/obs/flag"][0])
        assert bool(file["data/demo_20"].attrs["success"])
    if mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
        with h5py.File(tmp_path / "dataset_failed.hdf5") as file:
            assert set(file["data"]) == {"demo_10"}
            assert file["data/demo_10/actions"][()].tobytes() == values[1:2].numpy().tobytes()
            assert not bool(file["data/demo_10"].attrs["success"])


def test_recorder_storage_boundary(source_checkout_root):
    """Keep storage independent of consumers, with batched ownership and no Python world loops."""
    import isaaclab.utils.datasets.episode_recorder as module

    assert not (source_checkout_root / "source/isaaclab/isaaclab/managers/recorder_manager.py").exists()
    assert not (source_checkout_root / "source/isaaclab/isaaclab/envs/mdp/recorders/recorders_cfg.py").exists()
    import isaaclab.managers as managers
    from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnvCfg

    assert "recorders" not in ManagerBasedEnvCfg.__dataclass_fields__
    assert "final_obs_every_step" not in ManagerBasedRLEnvCfg.__dataclass_fields__
    import isaaclab.envs.mdp as mdp
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.envs.mdp import recorders
    from isaaclab.envs.mdp.recorders import recorders as capture_functions

    assert "record_step" not in mdp.__all__
    assert "record_step" not in recorders.__all__
    assert "record_step" not in vars(capture_functions)
    lifecycle_operations = {"process_action", "apply_action", "step", "_reset_idx", "export_episodes", "set_decimation"}
    capture_source = ast.parse(inspect.getsource(capture_functions))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in lifecycle_operations for node in ast.walk(capture_source)
    )
    assert "simulate" not in vars(ManagerBasedEnv)
    assert not {"compute_transition", "reset_done", "finish_step"} & vars(ManagerBasedRLEnv).keys()
    assert not {"RecorderManager", "RecorderManagerBaseCfg", "RecorderTerm", "RecorderTermCfg"} & set(managers.__all__)
    for package in ("isaaclab", "isaaclab_experimental", "isaaclab_tasks", "isaaclab_neural"):
        package_dir = source_checkout_root / "source" / package / package
        for path in package_dir.rglob("*env*.py"):
            tree = ast.parse(path.read_text())
            assert not any(
                isinstance(node, ast.Attribute) and node.attr == "recorder_manager" for node in ast.walk(tree)
            )

    source = ast.parse(inspect.getsource(module))
    # Batched buffers own collection; snapshots and capture terms have no second writer lifecycle.
    assert not {"add", "pre_export"} & vars(EpisodeData).keys()
    # Observation evaluation belongs to explicit terms, never generic capture dispatch.
    assert (
        not {"record_pre_step", "record_post_step", "_prepare_terms", "get_term"} & vars(module.EpisodeRecorder).keys()
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"observation_manager", "_episodes", "_env", "_terms", "termination_manager"}
        for node in ast.walk(source)
    )
    allowed = sys.stdlib_module_names | {"isaaclab", "torch", "warp"}
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            assert (node.module or "").split(".")[0] in allowed
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("exported_"):
            assert not any(isinstance(child, ast.Name) and child.id == "hasattr" for child in ast.walk(node))
    field = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "_RecorderField")
    append = next(node for node in field.body if isinstance(node, ast.FunctionDef) and node.name == "append")
    assert not any(isinstance(node, (ast.For, ast.While, ast.comprehension)) for node in ast.walk(append))
