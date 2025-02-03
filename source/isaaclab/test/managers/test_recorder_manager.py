# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

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

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
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
    return namedtuple("ManagerBasedEnv", ["num_envs", "device", "cfg", "termination_manager"])(
        20, device, dict(), dummy_termination_manager
    )


class TestRecorderManager(unittest.TestCase):
    """Test cases for various situations with recorder manager."""

    def setUp(self) -> None:
        self.dataset_dir = tempfile.mkdtemp()

    def tearDown(self):
        # delete the temporary directory after the test
        shutil.rmtree(self.dataset_dir)

    def create_dummy_recorder_manager_cfg(self) -> DummyRecorderManagerCfg:
        """Get the dummy recorder manager configurations."""
        cfg = DummyRecorderManagerCfg()
        cfg.dataset_export_dir_path = self.dataset_dir
        cfg.dataset_filename = f"{uuid.uuid4()}.hdf5"
        return cfg

    def test_str(self):
        """Test the string representation of the recorder manager."""
        # create recorder manager
        cfg = DummyRecorderManagerCfg()
        recorder_manager = RecorderManager(cfg, create_dummy_env())
        self.assertEqual(len(recorder_manager.active_terms), 2)
        # print the expected string
        print()
        print(recorder_manager)

    def test_initialize_dataset_file(self):
        """Test the initialization of the dataset file."""
        # create recorder manager
        cfg = self.create_dummy_recorder_manager_cfg()
        _ = RecorderManager(cfg, create_dummy_env())

        # check if the dataset is created
        self.assertTrue(os.path.exists(os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename)))

    def test_record(self):
        """Test the recording of the data."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                env = create_dummy_env(device)
                # create recorder manager
                recorder_manager = RecorderManager(self.create_dummy_recorder_manager_cfg(), env)

                # record the step data
                recorder_manager.record_pre_step()
                recorder_manager.record_post_step()

                recorder_manager.record_pre_step()
                recorder_manager.record_post_step()

                # check the recorded data
                for env_id in range(env.num_envs):
                    episode = recorder_manager.get_episode(env_id)
                    self.assertEqual(episode.data["record_pre_step"].shape, (2, 4))
                    self.assertEqual(episode.data["record_post_step"].shape, (2, 5))

                # Trigger pre-reset callbacks which then export and clean the episode data
                recorder_manager.record_pre_reset(env_ids=None)
                for env_id in range(env.num_envs):
                    episode = recorder_manager.get_episode(env_id)
                    self.assertTrue(episode.is_empty())

                recorder_manager.record_post_reset(env_ids=None)
                for env_id in range(env.num_envs):
                    episode = recorder_manager.get_episode(env_id)
                    self.assertEqual(episode.data["record_post_reset"].shape, (1, 3))


if __name__ == "__main__":
    run_tests()
