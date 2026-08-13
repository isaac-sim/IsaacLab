# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single-stage Franka Pour task configuration."""

import pytest

from isaaclab_tasks.contrib.franka_pour import pour_env
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    FrankaPourResetDatasetEnvCfg,
    _configure_mpm_capacities,
    _reset_dataset_task_contract,
    _resolve_pour_solver_tree,
)

_RESET_CONTRACT_FIELDS = {
    "robot_asset",
    "robot_physics_payload_sha256",
    "source_box_half",
    "source_mouth_height",
    "target_box_half",
    "target_rim_height",
    "cup_grasp_tcp_quat_c",
    "cup_grasp_height",
    "tcp_body_name",
    "tcp_offset_pos",
    "tcp_offset_rot",
    "gripper_open_pos",
    "gripper_grasp_reset_target",
    "gripper_contact_min_deflection",
    "collider_margin",
    "mpm_collider_margin",
    "particle_count",
    "particle_spacing",
}


def test_reset_dataset_path_is_repo_relative_and_missing_error_is_actionable(monkeypatch, tmp_path):
    """Relative artifacts share the generator's root and missing files name the recovery command."""
    repo_root = tmp_path / "repo"
    dataset_path = repo_root / "datasets/franka_pour/reset_dataset.pt"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.touch()
    monkeypatch.setattr(pour_env, "_REPO_ROOT", repo_root)
    monkeypatch.chdir(tmp_path)

    assert pour_env._resolve_reset_dataset_path("datasets/franka_pour/reset_dataset.pt") == dataset_path.resolve()

    dataset_path.unlink()
    with pytest.raises(FileNotFoundError) as exc_info:
        pour_env._resolve_reset_dataset_path("datasets/franka_pour/reset_dataset.pt")
    message = str(exc_info.value)
    assert str(dataset_path.resolve()) in message
    assert "uv run python scripts/tools/generate_franka_pour_reset_dataset.py --device cuda:0" in message


def test_config_is_fully_authored_once_with_compact_reset_contract():
    """Scene assets exist immediately and expose only reset-state compatibility fields."""
    cfg = FrankaPourResetDatasetEnvCfg()

    assert not hasattr(cfg, "finalize")
    assert cfg.scene.source_cup is not None
    assert cfg.scene.target_cup is not None
    assert cfg.scene.media is not None
    assert cfg.reset_dataset_content_sha256 is None
    task_contract = _reset_dataset_task_contract(cfg)
    assert task_contract["particle_count"] == 245
    assert set(task_contract) == _RESET_CONTRACT_FIELDS


def test_nested_overrides_are_authoritative_without_rebuilding_assets():
    """Hydra-style updates mutate the actual scene and solver configuration objects."""
    cfg = FrankaPourResetDatasetEnvCfg()
    source_cup = cfg.scene.source_cup
    media = cfg.scene.media
    solver = _resolve_pour_solver_tree(cfg)
    reset_contract = _reset_dataset_task_contract(cfg)

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
    assert tuple(cfg.scene.source_cup.init_state.pos) == (0.6, 0.0, 0.0)
    assert cfg.sim.physics.num_substeps == 5
    assert solver.media_solver.max_iterations == 17
    assert not cfg.sim.physics.use_cuda_graph
    assert _reset_dataset_task_contract(cfg) == reset_contract


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


def test_final_validation_checks_runtime_allocation_values():
    """Invalid post-construction overrides are rejected by the standard config hook."""
    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.mpm_cell_capacity_alignment = 0

    with pytest.raises(ValueError, match="mpm_cell_capacity_alignment must be a positive integer"):
        cfg.validate()


@pytest.mark.parametrize(
    ("term_path", "parameter", "value", "message"),
    [
        (("observations", "media", "cup_velocity"), "surface_radius", 0.0, "surface_radius"),
        (("terminations", "source_receiver_overlap"), "clearance", -0.1, "clearance"),
        (("terminations", "lost_grasp"), "terminate", 1, "lost_grasp.terminate"),
    ],
)
def test_constant_mdp_parameters_are_validated_before_stepping(term_path, parameter, value, message):
    """Manager-term constants fail at config validation instead of inside batched MDP calls."""
    cfg = FrankaPourResetDatasetEnvCfg()
    term = cfg
    for field in term_path:
        term = getattr(term, field)
    term.params[parameter] = value

    with pytest.raises((TypeError, ValueError), match=message):
        cfg.validate()
