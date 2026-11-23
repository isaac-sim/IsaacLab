# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pytest
import torch
import warp as wp
from gymnasium.vector import AutoresetMode

from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

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
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}
        self.active_episode_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
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
        assert torch.stack(episode.data["record_pre_step"]).shape == (2, 4)
        assert torch.stack(episode.data["record_post_step"]).shape == (2, 5)

    # Trigger pre-reset callbacks which then export and clean the episode data
    recorder_manager.record_pre_reset(env_ids=slice(None))
    for env_id in range(env.num_envs):
        episode = recorder_manager.get_episode(env_id)
        assert episode.is_empty()

    recorder_manager.record_post_reset(env_ids=slice(None))
    for env_id in range(env.num_envs):
        episode = recorder_manager.get_episode(env_id)
        assert torch.stack(episode.data["record_post_reset"]).shape == (1, 3)
    recorder_manager.reset(slice(1, None, 2))
    for env_id in range(env.num_envs):
        assert recorder_manager.get_episode(env_id).is_empty() == (env_id % 2 == 1)
    recorder_manager.close()


class EnvironmentIndexRecorderTerm(RecorderTerm):
    """Identify environment rows in initial, terminal, and full-batch step records."""

    def record_pre_reset(self, env_ids):
        assert len(env_ids) > 0, "Empty selections must not invoke recorder terms"
        return "terminal", torch.as_tensor(env_ids).reshape(-1, 1)

    def record_post_reset(self, env_ids):
        assert len(env_ids) > 0, "Empty selections must not invoke recorder terms"
        return "initial", torch.as_tensor(env_ids).reshape(-1, 1)

    def record_pre_step(self):
        return "pre", torch.arange(self._env.num_envs).reshape(-1, 1)

    def record_post_step(self):
        return "post", {"nested": {"index": torch.arange(self._env.num_envs).reshape(-1, 1)}}

    def record_post_physics_decimation_step(self):
        environment_indices = torch.arange(self._env.num_envs, dtype=torch.float32).reshape(-1, 1)
        return "physics", wp.from_torch(environment_indices)


def test_step_recording_selects_full_batch_rows(dataset_dir):
    """Selected rows retain their environment IDs and retired rows stay empty after export."""

    @configclass
    class SelectedEpisodeRecorderCfg(RecorderManagerBaseCfg):
        step = RecorderTermCfg(class_type=EnvironmentIndexRecorderTerm)

    wp.init()
    recorder_cfg = SelectedEpisodeRecorderCfg()
    recorder_cfg.dataset_export_dir_path = dataset_dir
    env = DummyEnv(num_envs=3)
    env.metadata["autoreset_mode"] = AutoresetMode.DISABLED
    env.active_episode_mask[:] = False
    recorder_manager = RecorderManager(recorder_cfg, env)
    selected_env_ids = torch.tensor([2, 0])
    recording_callbacks = (
        recorder_manager.record_pre_step,
        recorder_manager.record_post_step,
        recorder_manager.record_post_physics_decimation_step,
    )
    try:
        empty_env_ids = torch.empty(0, dtype=torch.int64)
        recorder_manager.record_pre_reset(empty_env_ids)
        recorder_manager.record_pre_reset(slice(None))
        assert recorder_manager.exported_failed_episode_count == 0
        env.active_episode_mask[selected_env_ids] = True
        recorder_manager.record_post_reset(selected_env_ids)
        recorder_manager.record_post_reset(empty_env_ids)
        for record_step in recording_callbacks:
            record_step(selected_env_ids)
            record_step(torch.empty(0, dtype=torch.int64))
        assert recorder_manager.get_episode(1).is_empty()
        for env_id in selected_env_ids.tolist():
            episode_data = recorder_manager.get_episode(env_id).data
            assert (
                len(episode_data["pre"])
                == len(episode_data["post"]["nested"]["index"])
                == len(episode_data["physics"])
                == 1
            )
            assert episode_data["pre"][0].item() == env_id
            assert episode_data["post"]["nested"]["index"][0].item() == env_id
            assert episode_data["physics"][0].item() == env_id
        recorder_manager.record_pre_reset([1, 2, 0])
        env.active_episode_mask[selected_env_ids] = False
        assert recorder_manager.exported_failed_episode_count == 2
        recorder_manager.record_pre_reset([0])
        recorder_manager.reset([0])
        env.active_episode_mask[0] = True
        recorder_manager.record_post_reset([0])
        for record_step in recording_callbacks:
            record_step([0])
        assert recorder_manager.get_episode(2).is_empty()
        assert not recorder_manager.get_episode(0).is_empty()

        # Resetting an environment whose episode already ended starts only its next recording.
        recorder_manager.record_pre_reset([2])
        recorder_manager.reset([2])
        env.active_episode_mask[2] = True
        recorder_manager.record_post_reset([2])
        assert recorder_manager.exported_failed_episode_count == 2
        assert list(recorder_manager.get_episode(2).data) == ["initial"]
        assert recorder_manager.get_episode(2).data["initial"][0].item() == 2
    finally:
        recorder_manager.close()
