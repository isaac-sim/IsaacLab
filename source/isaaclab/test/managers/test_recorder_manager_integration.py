# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for recorder behavior in a manager-based environment."""

from __future__ import annotations

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

from pathlib import Path

import h5py
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.integration


class DummyStepRecorderTerm(RecorderTerm):
    """Dummy recorder term that records data around each environment step."""

    def record_pre_step(self) -> tuple[str, torch.Tensor]:
        return "record_pre_step", torch.ones(self._env.num_envs, 4, device=self._env.device)

    def record_post_step(self) -> tuple[str, torch.Tensor]:
        return "record_post_step", torch.ones(self._env.num_envs, 5, device=self._env.device)


@configclass
class DummyRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configuration for environment lifecycle integration coverage."""

    @configclass
    class DummyStepRecorderTermCfg(RecorderTermCfg):
        """Configuration for the dummy step recorder term."""

        class_type: type[RecorderTerm] = DummyStepRecorderTerm

    record_step_term: DummyStepRecorderTermCfg = DummyStepRecorderTermCfg()
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL


def get_dataset_shapes(file_path: Path) -> dict[str, tuple[int, ...]]:
    """Read dataset shapes from an HDF5 file."""
    shapes = {}
    with h5py.File(file_path, "r") as file:

        def collect_shape(name, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[name] = obj.shape

        file.visititems(collect_shape)
    return shapes


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_manager_based_env_close_exports_buffered_recorder_data(device: str, tmp_path: Path):
    """Environment stepping and close should record and export buffered episodes."""
    sim_utils.create_new_stage()
    env_cfg = make_empty_manager_based_env_cfg(device=device, num_envs=2)
    env_cfg.recorders = DummyRecorderManagerCfg(
        export_in_close=True,
        dataset_export_dir_path=str(tmp_path),
        dataset_filename="manager_based_env_close.hdf5",
    )
    file_path = tmp_path / env_cfg.recorders.dataset_filename
    env = ManagerBasedEnv(cfg=env_cfg)
    num_steps = 3

    try:
        for _ in range(num_steps):
            env.step(torch.randn_like(env.action_manager.action))

        assert get_dataset_shapes(file_path) == {}
    finally:
        env.close()

    dataset_shapes = get_dataset_shapes(file_path)
    assert len(dataset_shapes) == 2 * env_cfg.scene.num_envs
    assert list(dataset_shapes.values()).count((num_steps, 4)) == env_cfg.scene.num_envs
    assert list(dataset_shapes.values()).count((num_steps, 5)) == env_cfg.scene.num_envs
