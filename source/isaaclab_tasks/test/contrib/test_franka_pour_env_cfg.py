# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single-stage Franka Pour task configuration."""

import pytest

from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    FrankaPourResetDatasetEnvCfg,
    _configure_mpm_capacities,
    _reset_dataset_task_contract,
    _resolve_pour_solver_tree,
)
from isaaclab_tasks.contrib.franka_pour.reset_dataset_io import reset_dataset_digest

_REFERENCE_TASK_CONTRACT_SHA256 = "804ce04acab78ee4dc1c9fceaad5364b735801e56406c67b3ffe2e027a113b21"
_REMOVED_MIRROR_FIELDS = {
    "arm_home",
    "cup_mass",
    "media_material",
    "mpm_iterations",
    "physics_substeps",
    "proxy_mass_scale",
    "voxel_size",
}


def test_config_is_fully_authored_once_with_unchanged_reset_contract():
    """Scene assets exist immediately and retain the reference artifact contract."""
    cfg = FrankaPourResetDatasetEnvCfg()

    assert not hasattr(cfg, "finalize")
    assert cfg.scene.source_cup is not None
    assert cfg.scene.target_cup is not None
    assert cfg.scene.media is not None
    assert _REMOVED_MIRROR_FIELDS.isdisjoint(cfg.to_dict())

    task_contract = _reset_dataset_task_contract(cfg)
    assert task_contract["particle_count"] == 245
    assert reset_dataset_digest(task_contract) == _REFERENCE_TASK_CONTRACT_SHA256


def test_nested_overrides_are_authoritative_without_rebuilding_assets():
    """Hydra-style updates mutate the actual scene and solver configuration objects."""
    cfg = FrankaPourResetDatasetEnvCfg()
    source_cup = cfg.scene.source_cup
    media = cfg.scene.media
    solver = _resolve_pour_solver_tree(cfg)

    cfg.from_dict(
        {
            "scene": {"source_cup": {"init_state": {"pos": [0.6, 0.0, 0.0]}}},
            "sim": {"physics": {"num_substeps": 5, "use_cuda_graph": False}},
        }
    )
    solver.media_solver.max_iterations = 17
    cfg.validate()

    assert cfg.scene.source_cup is source_cup
    assert cfg.scene.media is media
    assert _resolve_pour_solver_tree(cfg) == solver
    assert cfg.cup_reset_pos == (0.6, 0.0, 0.0)
    assert cfg.physics_substeps == 5
    assert cfg.mpm_iterations == 17
    assert not cfg.use_cuda_graph


def test_capacity_resolution_only_updates_world_dependent_solver_limits():
    """Late world-count resolution does not clone or reconstruct task assets."""
    cfg = FrankaPourResetDatasetEnvCfg()
    source_cup = cfg.scene.source_cup
    target_cup = cfg.scene.target_cup
    media = cfg.scene.media
    solver = _resolve_pour_solver_tree(cfg).media_solver

    cfg.scene.num_envs = 1
    _configure_mpm_capacities(cfg)
    assert (solver.max_active_cell_count, solver.max_leaf_node_count) == (512, -1)
    assert (solver.max_lower_node_count, solver.max_upper_node_count) == (32, 32)

    cfg.scene.num_envs = 7
    _configure_mpm_capacities(cfg)
    assert (solver.max_active_cell_count, solver.max_leaf_node_count) == (3584, -1)
    assert (solver.max_lower_node_count, solver.max_upper_node_count) == (224, 32)
    assert cfg.scene.source_cup is source_cup
    assert cfg.scene.target_cup is target_cup
    assert cfg.scene.media is media


def test_final_validation_checks_the_authoritative_solver_tree():
    """Invalid nested solver overrides are rejected by the standard config validation hook."""
    cfg = FrankaPourResetDatasetEnvCfg()
    _resolve_pour_solver_tree(cfg).media_solver.max_iterations = 0

    with pytest.raises(ValueError, match="mpm_iterations must be a positive integer"):
        cfg.validate()
