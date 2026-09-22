# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from pathlib import Path

import h5py
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.integration


def dummy_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Return a dummy observation."""
    return torch.randn((env.num_envs, 1), device=env.device)


@configclass
class ObservationWithHistoryCfg:
    """Single observation group with a history of 5 samples."""

    @configclass
    class GroupCfg(ObsGroup):
        history_length = 5
        dummy_term: ObsTerm = ObsTerm(func=dummy_observation)

    group: GroupCfg = GroupCfg()


class DummyStepRecorderTerm(RecorderTerm):
    """Recorder term that records data around each environment step."""

    def record_pre_step(self) -> tuple[str, torch.Tensor]:
        return "record_pre_step", torch.ones(self._env.num_envs, 4, device=self._env.device)

    def record_post_step(self) -> tuple[str, torch.Tensor]:
        return "record_post_step", torch.ones(self._env.num_envs, 5, device=self._env.device)


@configclass
class DummyRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configuration that exports every episode on close."""

    @configclass
    class DummyStepRecorderTermCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = DummyStepRecorderTerm

    record_step_term: DummyStepRecorderTermCfg = DummyStepRecorderTermCfg()
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    export_in_close: bool = True


def get_dataset_shapes(file_path: Path) -> dict[str, tuple[int, ...]]:
    """Read the dataset shapes from an HDF5 file."""
    shapes = {}
    with h5py.File(file_path, "r") as file:
        file.visititems(lambda name, obj: shapes.update({name: obj.shape}) if isinstance(obj, h5py.Dataset) else None)
    return shapes


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_step_with_empty_actions_and_observation_history(device):
    """An environment without action terms steps, and each step advances the observation history."""
    sim_utils.create_new_stage()
    env_cfg = make_empty_manager_based_env_cfg(device=device)
    env_cfg.observations = ObservationWithHistoryCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    try:
        assert env.action_manager.total_action_dim == 0
        assert env.action_manager.active_terms == []
        assert env.observation_manager.active_terms == {"group": ["dummy_term"]}
        assert env.observation_manager.group_obs_dim == {"group": (5,)}

        history = env.observation_manager._group_obs_term_history_buffer["group"]["dummy_term"]
        expected_length = torch.zeros((env.num_envs,), device=device, dtype=torch.int64)
        torch.testing.assert_close(history.current_length, expected_length)
        for _ in range(2):
            obs, _ = env.step(action=torch.zeros_like(env.action_manager.action))
            expected_length += 1
            torch.testing.assert_close(history.current_length, expected_length)
        assert obs["group"].shape == (env.num_envs, 5)
    finally:
        env.close()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_close_exports_buffered_recorder_data(device: str, tmp_path: Path):
    """Stepping records per-step data that is exported once the environment is closed."""
    sim_utils.create_new_stage()
    env_cfg = make_empty_manager_based_env_cfg(device=device, num_envs=2)
    env_cfg.recorders = DummyRecorderManagerCfg(
        dataset_export_dir_path=str(tmp_path), dataset_filename="manager_based_env_close.hdf5"
    )
    file_path = tmp_path / env_cfg.recorders.dataset_filename
    env = ManagerBasedEnv(cfg=env_cfg)
    num_steps = 3
    try:
        for _ in range(num_steps):
            env.step(torch.zeros_like(env.action_manager.action))
        assert get_dataset_shapes(file_path) == {}
    finally:
        env.close()

    dataset_shapes = get_dataset_shapes(file_path)
    assert len(dataset_shapes) == 2 * env_cfg.scene.num_envs
    assert list(dataset_shapes.values()).count((num_steps, 4)) == env_cfg.scene.num_envs
    assert list(dataset_shapes.values()).count((num_steps, 5)) == env_cfg.scene.num_envs
