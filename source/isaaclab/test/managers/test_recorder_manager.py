# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import ast
import inspect
import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import h5py
import pytest
import torch
import warp as wp

from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DummyResetRecorderTerm(RecorderTerm):
    """Dummy recorder term that records dummy data."""

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | None]:
        return "record_pre_reset", torch.ones(self._env.num_envs, 2, device=self._env.device)

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | None]:
        return "record_post_reset", torch.ones(self._env.num_envs, 3, device=self._env.device)


class DummyStepRecorderTerm(RecorderTerm):
    """Dummy recorder term that records dummy data."""

    def record_pre_step(self) -> tuple[str | None, torch.Tensor | None]:
        return "record_pre_step", torch.ones(self._env.num_envs, 4, device=self._env.device)

    def record_post_step(self) -> tuple[str | None, torch.Tensor | None]:
        return "record_post_step", torch.ones(self._env.num_envs, 5, device=self._env.device)


@configclass
class DummyRecorderManagerCfg(RecorderManagerBaseCfg):
    """Dummy recorder configurations."""

    @configclass
    class DummyResetRecorderTermCfg(RecorderTermCfg):
        """Configuration for the dummy reset recorder term."""

        class_type: type[RecorderTerm] = DummyResetRecorderTerm

    @configclass
    class DummyStepRecorderTermCfg(RecorderTermCfg):
        """Configuration for the dummy step recorder term."""

        class_type: type[RecorderTerm] = DummyStepRecorderTerm

    record_reset_term = DummyResetRecorderTermCfg()
    record_step_term = DummyStepRecorderTermCfg()

    dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class DummyEnvCfg:
    """Dummy environment configuration."""

    @configclass
    class DummySimCfg:
        """Configuration for the dummy sim."""

        dt: float = 0.01
        render_interval: int = 1

    @configclass
    class DummySceneCfg:
        """Configuration for the dummy scene."""

        num_envs: int = 1

    decimation: int = 1
    sim: DummySimCfg = DummySimCfg()
    scene: DummySceneCfg = DummySceneCfg()


class DummySimulation:
    """Minimal simulation double used by :class:`RecorderManager`."""

    def is_playing(self) -> bool:
        """Return whether the simulated timeline is playing."""
        return True


class DummyTerminationManager:
    """Minimal termination manager double without success terms."""

    active_terms: list[str] = []


class DummyEnv:
    """Minimal environment double used by recorder terms and metadata export."""

    def __init__(self, device: str = "cpu", num_envs: int = 20) -> None:
        self.num_envs = num_envs
        self.device = device
        self.sim = DummySimulation()
        self.cfg = DummyEnvCfg()
        self.cfg.scene.num_envs = num_envs
        self.termination_manager = DummyTerminationManager()


def create_dummy_env(device: str = "cpu", num_envs: int = 20) -> ManagerBasedEnv:
    """Create a minimal environment double."""

    return cast("ManagerBasedEnv", DummyEnv(device=device, num_envs=num_envs))


@pytest.fixture
def dataset_dir():
    """Create directory to dump results."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Cleanup
    shutil.rmtree(test_dir)


def test_str(dataset_dir):
    """Test the string representation of the recorder manager."""
    # create recorder manager
    cfg = DummyRecorderManagerCfg()
    cfg.dataset_export_dir_path = dataset_dir
    cfg.dataset_filename = f"{uuid.uuid4()}.hdf5"
    recorder_manager = RecorderManager(cfg, create_dummy_env())
    assert len(recorder_manager.active_terms) == 2
    manager_str = str(recorder_manager)
    assert "contains 2 active terms" in manager_str
    assert "record_reset_term" in manager_str
    assert "record_step_term" in manager_str
    # the dataset file is created on construction
    assert os.path.exists(os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename))
    recorder_manager.close()


def test_record(dataset_dir):
    """Test the recording of the data."""
    env = create_dummy_env()
    # create recorder manager
    cfg = DummyRecorderManagerCfg()
    cfg.dataset_export_dir_path = dataset_dir
    cfg.dataset_filename = f"{uuid.uuid4()}.hdf5"
    recorder_manager = RecorderManager(cfg, env)

    # record the step data
    recorder_manager.record_pre_step()
    recorder_manager.record_post_step()

    recorder_manager.record_pre_step()
    recorder_manager.record_post_step()

    # check the recorded data
    for env_id in range(env.num_envs):
        episode = recorder_manager.get_episode(env_id)
        assert episode.data["record_pre_step"].shape == (2, 4)
        assert episode.data["record_post_step"].shape == (2, 5)

    # Trigger pre-reset callbacks which then export and clean the episode data
    recorder_manager.record_pre_reset(env_ids=slice(None))
    for env_id in range(env.num_envs):
        episode = recorder_manager.get_episode(env_id)
        assert episode.is_empty()

    recorder_manager.record_post_reset(env_ids=slice(None))
    for env_id in range(env.num_envs):
        episode = recorder_manager.get_episode(env_id)
        assert episode.data["record_post_reset"].shape == (1, 3)
    recorder_manager.reset(slice(1, None, 2))
    for env_id in range(env.num_envs):
        assert recorder_manager.get_episode(env_id).is_empty() == (env_id % 2 == 1)
    recorder_manager.close()


def test_batched_episode_indices_and_ownership(monkeypatch):
    """Asynchronous resets, field cadence, growth and snapshots must preserve exact samples."""
    cfg = DummyRecorderManagerCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE, buffer_initial_capacity=2)
    manager = RecorderManager(cfg, create_dummy_env(num_envs=3))
    manager.add_to_episodes("unused", torch.empty(0, 7), [])
    assert not manager._fields
    with pytest.raises(ValueError, match="requires 3 rows"):
        manager.add_to_episodes("invalid", torch.empty(2, 7))
    assert not manager._fields
    source = torch.tensor([[1.0, -0.0], [2.0, float("nan")], [3.0, 4.0]])
    manager.add_to_episodes("obs", {"state": source})
    wp.init()
    manager.add_to_episodes("initial_state", wp.from_torch(torch.tensor([[10], [20], [30]], dtype=torch.int32)))
    source.fill_(99)
    manager.add_to_episodes("obs/state", source)
    snapshot = manager.get_episode(1)
    assert snapshot.data["obs"]["state"][0, 0] == 2
    assert torch.isnan(snapshot.data["obs"]["state"][0, 1])
    assert torch.signbit(manager.get_episode(0).data["obs"]["state"][0, 1])
    assert snapshot.data["initial_state"].dtype == torch.int32
    assert snapshot.success is None

    manager.reset(torch.tensor([1]))
    manager.add_to_episodes("obs/state", torch.tensor([[42.0, 43.0], [52.0, 53.0]]), [2, 1])
    manager.add_to_episodes("obs/state", torch.tensor([[62.0, 63.0]]), slice(0, 1))
    assert torch.equal(manager.get_episode(1).data["obs"]["state"], torch.tensor([[52.0, 53.0]]))
    assert manager.get_episode(1).data.keys() == {"obs"}
    assert torch.equal(manager.get_episode(2).data["obs"]["state"][:, 0], torch.tensor([3.0, 99.0, 42.0]))
    assert torch.equal(manager.get_episode(0).data["obs"]["state"][:, 0], torch.tensor([1.0, 99.0, 62.0]))
    assert snapshot.data["obs"]["state"].shape == (2, 2)
    snapshot = manager.get_episode(1)
    snapshot.data["obs"]["state"].fill_(-100)
    assert manager.get_episode(1).data["obs"]["state"][0, 0] == 52

    with pytest.raises(ValueError, match="changed shape"):
        manager.add_to_episodes("obs/state", torch.zeros(3, 5))
    with pytest.raises(ValueError, match="requires 3 rows"):
        manager.add_to_episodes("obs/state", torch.zeros(2, 2))
    with pytest.raises(ValueError, match="conflicts"):
        manager.add_to_episodes("obs", torch.zeros(3, 2))
    manager.add_to_episodes("obs/state", torch.empty(0, 2), [])
    assert manager.get_episode(0).data["obs"]["state"].shape == (3, 2)

    manager.reset()
    # A new schema is allowed once every world has released this field, e.g. manual annotation.
    manager.add_to_episodes("obs/state", torch.zeros(3, 5, dtype=torch.bool))
    assert manager.get_episode(0).data["obs"]["state"].dtype == torch.bool
    manager.reset()

    def forbidden(*args, **kwargs):
        raise AssertionError("Steady-state recording must reuse storage without episode objects or host index reads")

    # Architectural/performance gate: preallocated insertion and reset need neither episode objects,
    # per-world container appends, nor device-to-host scalar reads. Export-none also stays batched.
    with monkeypatch.context() as patch:
        patch.setattr(EpisodeData, "__init__", forbidden)
        patch.setattr(EpisodeData, "add", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch, "empty", forbidden)
        manager.add_to_episodes("obs/state", torch.zeros(3, 5, dtype=torch.bool))
        manager.set_success_to_episodes([0, 2], torch.tensor([[True], [False]]))
        manager.export_episodes([0, 2])
        manager.reset([1])
    assert manager.exported_successful_episode_count == 1
    assert manager.exported_failed_episode_count == 1
    assert all(manager.get_episode(i).is_empty() for i in range(3))
    manager.close()


@pytest.mark.parametrize("mode", list(DatasetExportMode))
def test_export_batched_episodes(dataset_dir, mode):
    """Export modes, custom IDs, nested tensor schema and counters survive partial finalization."""
    cfg = DummyRecorderManagerCfg(dataset_export_dir_path=dataset_dir, dataset_export_mode=mode, export_in_close=True)
    manager = RecorderManager(cfg, create_dummy_env(num_envs=3))
    values = torch.tensor([[1.0, -0.0], [2.0, float("nan")], [3.0, 4.0]])
    manager.add_to_episodes("actions", values)
    manager.add_to_episodes("obs", {"flag": torch.tensor([True, False, True])})
    manager.set_success_to_episodes(None, torch.tensor([[True], [False], [True]]))
    with pytest.raises(ValueError, match="must match"):
        manager.export_episodes([2, 1], [30])
    with pytest.raises(ValueError, match="must be unique"):
        manager.export_episodes([2, 1], [30, 30])
    manager.export_episodes(torch.tensor([2, 1]), [30, 10])
    assert manager.get_episode(1).is_empty()
    assert manager.get_episode(2).is_empty()
    assert manager.get_episode(0).success is True
    manager.export_episodes(slice(0, 1), [20])
    manager.export_episodes()  # Empty episodes do not affect counters.
    assert manager.exported_successful_episode_count == 2
    assert manager.exported_failed_episode_count == 1
    manager.close()
    if mode == DatasetExportMode.EXPORT_NONE:
        assert os.listdir(dataset_dir) == []
        return
    with h5py.File(os.path.join(dataset_dir, "dataset.hdf5")) as file:
        expected = {"demo_30", "demo_20"}
        if mode == DatasetExportMode.EXPORT_ALL:
            expected.add("demo_10")
        assert set(file["data"]) == expected
        assert file["data/demo_20/actions"][()].tobytes() == values[0:1].numpy().tobytes()
        assert bool(file["data/demo_30/obs/flag"][0])
        assert bool(file["data/demo_20"].attrs["success"])
    if mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
        with h5py.File(os.path.join(dataset_dir, "dataset_failed.hdf5")) as file:
            assert set(file["data"]) == {"demo_10"}
            assert file["data/demo_10/actions"][()].tobytes() == values[1:2].numpy().tobytes()
            assert not bool(file["data/demo_10"].attrs["success"])


def test_recorder_storage_boundary():
    """Do not reintroduce per-world episode ownership or Python world loops in the append path."""
    import isaaclab.managers.recorder_manager as module

    source = ast.parse(inspect.getsource(module))
    assert not any(isinstance(node, ast.Attribute) and node.attr == "_episodes" for node in ast.walk(source))
    forbidden = ("physicsnemo", "supermimic", "isaaclab_physicsnemo")
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(forbidden)
    field = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "_RecorderField")
    append = next(node for node in field.body if isinstance(node, ast.FunctionDef) and node.name == "append")
    assert not any(isinstance(node, (ast.For, ast.While, ast.comprehension)) for node in ast.walk(append))
