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

import h5py
import pytest
import torch

from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    import numpy as np

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


def get_file_contents(file_name: str, num_steps: int) -> dict[str, np.ndarray]:
    """Retrieves the contents of the hdf5 file
    Args:
        file_name: absolute path to the hdf5 file
        num_steps: number of steps taken in the environment
    Returns:
        dict[str, np.ndarray]: dictionary where keys are HDF5 paths and values are the corresponding data arrays.
    """
    data = {}
    with h5py.File(file_name, "r") as f:

        def get_data(name, obj):
            if isinstance(obj, h5py.Dataset):
                if "record_post_step" in name:
                    assert obj[()].shape == (num_steps, 5)
                elif "record_pre_step" in name:
                    assert obj[()].shape == (num_steps, 4)
                else:
                    raise Exception(f"The hdf5 file contains an unexpected data path, {name}")
                data[name] = obj[()]

        f.visititems(get_data)
    return data


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
    recorder_manager.close()


def test_initialize_dataset_file(dataset_dir):
    """Test the initialization of the dataset file."""
    # create recorder manager
    cfg = DummyRecorderManagerCfg()
    cfg.dataset_export_dir_path = dataset_dir
    cfg.dataset_filename = f"{uuid.uuid4()}.hdf5"
    recorder_manager = RecorderManager(cfg, create_dummy_env())

    # check if the dataset is created
    assert os.path.exists(os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename))
    recorder_manager.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_record(device, dataset_dir):
    """Test the recording of the data."""
    env = create_dummy_env(device=device)
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
    recorder_manager.record_pre_reset(env_ids=None)
    for env_id in range(env.num_envs):
        episode = recorder_manager.get_episode(env_id)
        assert episode.is_empty()

    recorder_manager.record_post_reset(env_ids=None)
    for env_id in range(env.num_envs):
        episode = recorder_manager.get_episode(env_id)
        assert torch.stack(episode.data["record_post_reset"]).shape == (1, 3)
    recorder_manager.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_close(device, dataset_dir):
    """Test whether close exports buffered data when ``export_in_close`` is enabled."""
    env = create_dummy_env(device=device, num_envs=2)
    cfg = DummyRecorderManagerCfg()
    cfg.export_in_close = True
    cfg.dataset_export_dir_path = dataset_dir
    cfg.dataset_filename = f"{uuid.uuid4()}.hdf5"
    recorder_manager = RecorderManager(cfg, env)
    num_steps = 3
    for _ in range(num_steps):
        recorder_manager.record_pre_step()
        recorder_manager.record_post_step()

    # check contents of hdf5 file
    file_name = os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename)
    data_pre_close = get_file_contents(file_name, num_steps)
    assert len(data_pre_close) == 0
    recorder_manager.close()
    data_post_close = get_file_contents(file_name, num_steps)
    assert len(data_post_close.keys()) == 2 * env.num_envs
