# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
import torch
import uuid
from collections import namedtuple
from collections.abc import Sequence

import pytest

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


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


def create_dummy_env(device: str = "cpu") -> ManagerBasedEnv:
    """Create a dummy environment."""

    class DummyTerminationManager:
        active_terms = []

    dummy_termination_manager = DummyTerminationManager()
    sim = SimulationContext()
    return namedtuple("ManagerBasedEnv", ["num_envs", "device", "sim", "cfg", "termination_manager"])(
        20, device, sim, dict(), dummy_termination_manager
    )


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


def test_record(dataset_dir):
    """Test the recording of the data."""
    for device in ("cuda:0", "cpu"):
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
            assert episode.data["record_pre_step"].shape == (2, 4)
            assert episode.data["record_post_step"].shape == (2, 5)

        # Trigger pre-reset callbacks which then export and clean the episode data
        recorder_manager.record_pre_reset(env_ids=None)
        for env_id in range(env.num_envs):
            episode = recorder_manager.get_episode(env_id)
            assert episode.is_empty()

        recorder_manager.record_post_reset(env_ids=None)
        for env_id in range(env.num_envs):
            episode = recorder_manager.get_episode(env_id)
            assert episode.data["record_post_reset"].shape == (1, 3)
