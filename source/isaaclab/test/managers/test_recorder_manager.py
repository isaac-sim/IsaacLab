# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os
import shutil
import tempfile
import uuid
from collections import namedtuple
from collections.abc import Sequence
from typing import TYPE_CHECKING

import h5py
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

if TYPE_CHECKING:
    import numpy as np


class DummyResetRecorderTerm(RecorderTerm):
    """Dummy recorder term that records dummy data."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | None]:
        return "record_pre_reset", torch.ones(self._env.num_envs, 2, device=self._env.device)

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | None]:
        return "record_post_reset", torch.ones(self._env.num_envs, 3, device=self._env.device)


class DummyStepRecorderTerm(RecorderTerm):
    """Dummy recorder term that records dummy data."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

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
class EmptyManagerCfg:
    """Empty manager specifications for the environment."""

    pass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


def get_empty_base_env_cfg(device: str = "cuda", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""

    @configclass
    class EmptyEnvCfg(ManagerBasedEnvCfg):
        """Configuration for the empty test environment."""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()
        recorders: EmptyManagerCfg = EmptyManagerCfg()

        def __post_init__(self):
            """Post initialization."""
            # step settings
            self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
            # simulation settings
            self.sim.dt = 0.005  # sim step every 5ms: 200Hz
            self.sim.render_interval = self.decimation  # render every 4 sim steps
            # pass device down from test
            self.sim.device = device

    return EmptyEnvCfg()


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

        dt = 0.01
        render_interval = 1

    @configclass
    class DummySceneCfg:
        """Configuration for the dummy scene."""

        num_envs = 1

    decimation = 1
    sim = DummySimCfg()
    scene = DummySceneCfg()


def create_dummy_env(device: str = "cpu") -> ManagerBasedEnv:
    """Create a dummy environment."""

    class DummyTerminationManager:
        active_terms = []

    dummy_termination_manager = DummyTerminationManager()
    sim = SimulationContext()
    dummy_cfg = DummyEnvCfg()

    return namedtuple("ManagerBasedEnv", ["num_envs", "device", "sim", "cfg", "termination_manager"])(
        20, device, sim, dummy_cfg, dummy_termination_manager
    )


@pytest.fixture
def dataset_dir():
    """Create directory to dump results."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Cleanup
    shutil.rmtree(test_dir)


@pytest.fixture(autouse=True)
def cleanup_simulation_context():
    """Fixture to ensure SimulationContext is cleared after each test."""
    yield
    # Cleanup after test
    SimulationContext.clear_instance()


def test_str(dataset_dir):
    """Test the string representation of the recorder manager."""
    # create recorder manager
    cfg = DummyRecorderManagerCfg()
    recorder_manager = RecorderManager(cfg, create_dummy_env())
    assert len(recorder_manager.active_terms) == 2
    # print the expected string
    print(recorder_manager)


def test_initialize_dataset_file(dataset_dir):
    """Test the initialization of the dataset file."""
    # create recorder manager
    cfg = DummyRecorderManagerCfg()
    cfg.dataset_export_dir_path = dataset_dir
    cfg.dataset_filename = f"{uuid.uuid4()}.hdf5"
    _ = RecorderManager(cfg, create_dummy_env())

    # check if the dataset is created
    assert os.path.exists(os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename))


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_record(device, dataset_dir):
    """Test the recording of the data."""
    env = create_dummy_env(device)
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


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_close(device, dataset_dir):
    """Test whether data is correctly exported in the close function when fully integrated with ManagerBasedEnv and
    `export_in_close` is True."""
    # create a new stage
    sim_utils.create_new_stage()
    # create environment
    env_cfg = get_empty_base_env_cfg(device=device, num_envs=2)
    cfg = DummyRecorderManagerCfg()
    cfg.export_in_close = True
    cfg.dataset_export_dir_path = dataset_dir
    cfg.dataset_filename = f"{uuid.uuid4()}.hdf5"
    env_cfg.recorders = cfg
    env = ManagerBasedEnv(cfg=env_cfg)
    num_steps = 3
    for _ in range(num_steps):
        act = torch.randn_like(env.action_manager.action)
        obs, ext = env.step(act)
    # check contents of hdf5 file
    file_name = f"{env_cfg.recorders.dataset_export_dir_path}/{env_cfg.recorders.dataset_filename}"
    data_pre_close = get_file_contents(file_name, num_steps)
    assert len(data_pre_close) == 0
    env.close()
    data_post_close = get_file_contents(file_name, num_steps)
    assert len(data_post_close.keys()) == 2 * env_cfg.scene.num_envs
