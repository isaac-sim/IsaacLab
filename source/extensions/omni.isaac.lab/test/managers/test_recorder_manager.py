# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os
import shutil
import tempfile
import torch
import unittest
import uuid
from collections import namedtuple
from collections.abc import Sequence

from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import (
    DatasetExportMode,
    RecorderManager,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from omni.isaac.lab.utils import configclass


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


class TestRecorderManager(unittest.TestCase):
    """Test cases for various situations with recorder manager."""

    def setUp(self) -> None:
        self.dataset_dir = tempfile.mkdtemp()
        self.dataset_file_name = f"{uuid.uuid4()}.hdf5"

        # set up the environment
        self.num_envs = 20
        self.device = "cpu"

        class DummyTerminationManager:
            active_terms = []

        self.dummy_termination_manager = DummyTerminationManager()
        # create dummy environment
        self.env = namedtuple("ManagerBasedEnv", ["num_envs", "device", "cfg", "termination_manager"])(
            self.num_envs, self.device, dict(), self.dummy_termination_manager
        )

    def tearDown(self):
        # delete the temporary directory after the test
        shutil.rmtree(self.dataset_dir)

    def test_str(self):
        """Test the string representation of the recorder manager."""
        # create recorder manager
        cfg = DummyRecorderManagerCfg()
        cfg.dataset_export_dir_path = self.dataset_dir
        cfg.dataset_filename = self.dataset_file_name
        recorder_manager = RecorderManager(cfg, self.env)
        self.assertEqual(len(recorder_manager.active_terms), 2)
        # print the expected string
        print()
        print(recorder_manager)

    def test_initialize_dataset_file(self):
        """Test the initialization of the dataset file."""
        # create recorder manager
        cfg = DummyRecorderManagerCfg()
        cfg.dataset_export_dir_path = self.dataset_dir
        cfg.dataset_filename = self.dataset_file_name
        _ = RecorderManager(cfg, self.env)

        # check if the dataset is created
        self.assertTrue(os.path.exists(os.path.join(self.dataset_dir, self.dataset_file_name)))

    def test_record(self):
        """Test the recording of the data."""
        # create recorder manager
        recorder_manager = RecorderManager(DummyRecorderManagerCfg(), self.env)

        # record the step data
        recorder_manager.record_pre_step()
        recorder_manager.record_post_step()

        recorder_manager.record_pre_step()
        recorder_manager.record_post_step()

        # check the recorded data
        for env_id in range(self.num_envs):
            episode = recorder_manager.get_episode(env_id)
            self.assertEqual(episode.data["record_pre_step"].shape, (2, 4))
            self.assertEqual(episode.data["record_post_step"].shape, (2, 5))

        # Trigger pre-reset callbacks which then export and clean the episode data
        recorder_manager.record_pre_reset(env_ids=None)
        for env_id in range(self.num_envs):
            episode = recorder_manager.get_episode(env_id)
            self.assertTrue(episode.is_empty())

        recorder_manager.record_post_reset(env_ids=None)
        for env_id in range(self.num_envs):
            episode = recorder_manager.get_episode(env_id)
            self.assertEqual(episode.data["record_post_reset"].shape, (1, 3))


if __name__ == "__main__":
    run_tests()
