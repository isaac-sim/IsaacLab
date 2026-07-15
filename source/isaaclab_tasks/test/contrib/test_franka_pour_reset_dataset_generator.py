# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused CPU tests for the Franka Pour reset-dataset cache."""

import math
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab_tasks.contrib.franka_pour.reset_dataset_generator import (
    FRANKA_POUR_RESET_DATASET_TASK_ID,
    FrankaPourResetDatasetGenerator,
    FrankaPourResetDatasetGeneratorCfg,
    above_target_tilted_mask,
    build_reset_dataset_payload,
    grasp_objective_components,
    normalize_grasp_objectives,
    oriented_box_supported_by_bounds,
    oriented_boxes_overlap,
    reset_dataset_content_sha256,
    select_production_reset_rows,
    source_root_position_from_tcp_grasp,
    validate_production_reset_dataset,
    validate_reset_dataset,
)


def _identity_poses(positions: torch.Tensor) -> torch.Tensor:
    poses = torch.zeros((positions.shape[0], 7), dtype=positions.dtype)
    poses[:, :3] = positions
    poses[:, 6] = 1.0
    return poses


def _tiny_states() -> dict[str, torch.Tensor]:
    """Build two grasping and two non-grasping states with valid cache semantics."""
    count = 4
    source_pose = _identity_poses(
        torch.tensor(
            (
                (0.50, 0.00, 0.10),
                (0.55, 0.05, 0.20),
                (0.48, -0.12, 0.00),
                (0.62, 0.10, 0.00),
            ),
            dtype=torch.float32,
        )
    )
    target_pose = _identity_poses(
        torch.tensor(
            (
                (0.62, 0.00, 0.00),
                (0.56, 0.04, 0.00),
                (0.65, 0.10, 0.00),
                (0.50, -0.18, 0.00),
            ),
            dtype=torch.float32,
        )
    )
    return {
        "arm_joint_position": torch.linspace(-0.5, 0.5, count * 7, dtype=torch.float32).reshape(count, 7),
        "arm_joint_velocity": torch.zeros((count, 7), dtype=torch.float32),
        "finger_joint_position": torch.tensor(
            ((0.028, 0.028), (0.028, 0.028), (0.000, 0.000), (0.040, 0.040)), dtype=torch.float32
        ),
        "finger_joint_velocity": torch.zeros((count, 2), dtype=torch.float32),
        "finger_joint_target": torch.tensor(
            ((0.021, 0.021), (0.021, 0.021), (0.000, 0.000), (0.040, 0.040)), dtype=torch.float32
        ),
        "source_root_pose": source_pose,
        "source_root_velocity": torch.zeros((count, 6), dtype=torch.float32),
        "target_root_pose": target_pose,
        "target_root_velocity": torch.zeros((count, 6), dtype=torch.float32),
        "category": torch.tensor((1, 1, 0, 0), dtype=torch.int8),
        "objective": torch.tensor((0.0, 1.0, -1.0, -1.0), dtype=torch.float32),
        "objective_raw": torch.tensor((0.2, 0.8, -1.0, -1.0), dtype=torch.float32),
        "objective_components": torch.tensor(
            ((0.1, 0.2, 0.3), (0.8, 0.7, 0.9), (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)),
            dtype=torch.float32,
        ),
        "grasp_region": torch.tensor((0, 0, -1, -1), dtype=torch.int8),
        "grasp_side": torch.tensor((0, 3, -1, -1), dtype=torch.int8),
        "attempt_id": torch.arange(count, dtype=torch.int64),
        "particle_layout_id": torch.zeros(count, dtype=torch.int32),
        "ik_cost": torch.tensor((1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5), dtype=torch.float32),
        "ik_position_residual": torch.tensor((1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4), dtype=torch.float32),
        "ik_rotation_residual": torch.tensor((1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3), dtype=torch.float32),
    }


def _tiny_metadata() -> dict:
    return {
        "seed": 7,
        "state_count": 4,
        "category_names": ("non_grasping", "grasping"),
        "category_counts": torch.tensor((2, 2), dtype=torch.int64),
        "joint_names": tuple(f"panda_joint{index}" for index in range(1, 8)),
        "frame": "environment",
        "quaternion_order": "xyzw",
        "particle_solver_state": "fresh_zero",
        "source_region_center": torch.tensor((0.5, 0.0, 0.0595), dtype=torch.float32),
        "objective_weights": torch.full((3,), 1.0 / 3.0, dtype=torch.float32),
        "objective_component_names": (
            "source_distance",
            "target_gated_inversion",
            "target_alignment",
        ),
        "objective_raw_min_max": torch.tensor((0.2, 0.8), dtype=torch.float32),
        "attempt_counts": torch.tensor((3, 4), dtype=torch.int64),
        "rejection_counts": {"collision": torch.tensor((1, 2), dtype=torch.int64)},
        "sampling_and_validation_config": {"central_workspace_fraction": 0.9},
        "task_contract": {
            "source_box_half": (0.028, 0.028, 0.0595),
            "gripper_position_range": (0.0, 0.04),
            "cup_grasp_height": 0.083,
            "gripper_preload_pos": 0.024,
            "gripper_grasp_reset_target": 0.021,
            "gripper_contact_min_deflection": 0.001,
        },
    }


def _tiny_payload(
    states: dict[str, torch.Tensor] | None = None,
    metadata: dict | None = None,
) -> dict:
    particle_local_positions = torch.tensor(
        (((-0.01, -0.01, 0.01), (0.01, -0.01, 0.01), (0.0, 0.01, 0.02)),), dtype=torch.float32
    )
    return build_reset_dataset_payload(
        _tiny_states() if states is None else states,
        particle_local_positions,
        _tiny_metadata() if metadata is None else metadata,
        FrankaPourResetDatasetGeneratorCfg(
            grasping_count=2,
            non_grasping_count=2,
            near_pour_grasp_count=0,
            batch_size=4,
        ),
    )


def _tiny_production_payload() -> dict:
    """Add the provenance emitted by dynamic validation to a tiny payload."""
    payload = _tiny_payload()
    payload["metadata"]["dynamic_validation"] = {
        "source_content_sha256": "0" * 64,
        "steps": 60,
        "settle_steps": 8,
        "failure_dwell_steps": 2,
        "failure_counts": {"nonfinite": 0, "grasp_lost": 1},
        "balance_trimmed": 0,
    }
    payload["content_sha256"] = reset_dataset_content_sha256(payload)
    return payload


def _validate_tiny_production_payload(payload: dict) -> None:
    """Validate provenance while retaining the compact test fixture quotas."""
    validate_production_reset_dataset(
        payload,
        expected_grasping_count=2,
        expected_non_grasping_count=2,
    )


def test_grasp_objective_components_match_known_geometric_states():
    half_sqrt_two = math.sqrt(0.5)
    source_pose = torch.tensor(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.15, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0),
            (0.075, 0.0, 0.075, half_sqrt_two, 0.0, 0.0, half_sqrt_two),
        ),
        dtype=torch.float64,
    )
    target_pose = _identity_poses(
        torch.tensor(((1.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=torch.float64)
    )

    components = grasp_objective_components(
        source_pose,
        target_pose,
        source_region_center=(0.0, 0.0, 0.0),
        cup_center_offset=(0.0, 0.0, 0.0),
        target_rim_height=0.0,
    )

    expected = torch.tensor(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (math.sqrt(0.5), 0.125, 0.25)),
        dtype=torch.float64,
    )
    torch.testing.assert_close(components, expected, rtol=1.0e-12, atol=1.0e-12)


def test_inversion_credit_is_zero_away_from_target():
    source_pose = torch.tensor(((0.10, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0),), dtype=torch.float64)
    target_pose = _identity_poses(torch.tensor(((0.0, 0.0, 0.0),), dtype=torch.float64))

    components = grasp_objective_components(
        source_pose,
        target_pose,
        source_region_center=(0.0, 0.0, 0.0),
        cup_center_offset=(0.0, 0.0, 0.0),
        target_rim_height=0.05,
        inversion_gate_horizontal_threshold=0.07,
    )

    assert components[0, 0] > 0.0
    assert components[0, 1] == 0.0
    assert components[0, 2] > 0.0


def test_grasp_objective_components_transform_cup_center_offset():
    source_pose = _identity_poses(torch.zeros((1, 3), dtype=torch.float64))
    target_pose = _identity_poses(torch.zeros((1, 3), dtype=torch.float64))

    components = grasp_objective_components(
        source_pose,
        target_pose,
        source_region_center=(0.0, 0.0, 0.1),
        cup_center_offset=(0.0, 0.0, 0.1),
        target_rim_height=0.025,
    )

    torch.testing.assert_close(components, torch.tensor(((0.0, 0.0, 0.5),), dtype=torch.float64))


def test_source_root_position_seats_configured_grasp_point_at_tcp_not_cup_center():
    half_sqrt_two = math.sqrt(0.5)
    tcp_position = torch.tensor(((0.4, -0.1, 0.3),), dtype=torch.float64)
    tcp_quaternion = torch.tensor(((0.0, 0.0, 0.0, 1.0),), dtype=torch.float64)
    source_quaternion = torch.tensor(((0.0, half_sqrt_two, 0.0, half_sqrt_two),), dtype=torch.float64)
    grasp_offset = torch.tensor((0.0, 0.0, 0.083), dtype=torch.float64)
    seating_offset = torch.tensor(((0.001, 0.0, -0.002),), dtype=torch.float64)

    source_position = source_root_position_from_tcp_grasp(
        tcp_position,
        tcp_quaternion,
        source_quaternion,
        grasp_offset,
        seating_offset,
    )
    restored_grasp_position = source_position + torch.nn.functional.normalize(
        source_quaternion,
        dim=-1,
    ).new_tensor(((0.083, 0.0, 0.0),))

    torch.testing.assert_close(restored_grasp_position, tcp_position + seating_offset)


def test_normalize_grasp_objectives_maps_global_extrema_to_unit_interval():
    raw = torch.tensor((2.0, 4.0, 8.0), dtype=torch.float64)

    normalized = normalize_grasp_objectives(raw)

    torch.testing.assert_close(normalized, torch.tensor((0.0, 1.0 / 3.0, 1.0), dtype=torch.float64))
    assert normalized.dtype == raw.dtype
    assert normalized.device == raw.device


def test_normalize_grasp_objectives_rejects_degenerate_range():
    with pytest.raises(ValueError):
        normalize_grasp_objectives(torch.full((3,), 0.4))


def test_sampler_defaults_define_the_required_exact_twenty_thousand_states():
    cfg = FrankaPourResetDatasetGeneratorCfg()

    assert cfg.grasping_count == 10_000
    assert cfg.non_grasping_count == 10_000
    assert cfg.grasping_count + cfg.non_grasping_count == 20_000


def test_sampler_runtime_interface_accepts_oversampled_candidate_quotas(monkeypatch):
    cfg = FrankaPourResetDatasetGeneratorCfg(grasping_count=8, non_grasping_count=4, near_pour_grasp_count=4)
    env = SimpleNamespace(cfg=object(), device="cpu")
    monkeypatch.setattr(FrankaPourResetDatasetGenerator, "_build_ik_context", lambda _self: None)

    sampler = FrankaPourResetDatasetGenerator(env, cfg)

    assert sampler.cfg.grasping_count == 8


@pytest.mark.parametrize("near_pour_count", (0, 8))
def test_sampler_runtime_interface_requires_both_grasp_regions(near_pour_count):
    cfg = FrankaPourResetDatasetGeneratorCfg(
        grasping_count=8,
        non_grasping_count=4,
        near_pour_grasp_count=near_pour_count,
    )

    with pytest.raises(ValueError, match="requires both broad and near-pour"):
        FrankaPourResetDatasetGenerator(None, cfg)


def test_sampler_runtime_interface_rejects_unbalanced_grasp_region_quota():
    cfg = FrankaPourResetDatasetGeneratorCfg(grasping_count=8, non_grasping_count=4, near_pour_grasp_count=2)

    with pytest.raises(ValueError, match="near_pour_grasp_count must be divisible by four"):
        FrankaPourResetDatasetGenerator(None, cfg)


def test_production_row_selection_enforces_exact_balanced_quotas():
    non_grasping_count = 10_002
    broad_per_side = 2_252
    near_per_side = 252
    category = [torch.zeros(non_grasping_count, dtype=torch.int8)]
    region = [torch.full((non_grasping_count,), -1, dtype=torch.int8)]
    side = [torch.full((non_grasping_count,), -1, dtype=torch.int8)]
    for region_id, per_side in ((0, broad_per_side), (1, near_per_side)):
        for side_id in range(4):
            category.append(torch.ones(per_side, dtype=torch.int8))
            region.append(torch.full((per_side,), region_id, dtype=torch.int8))
            side.append(torch.full((per_side,), side_id, dtype=torch.int8))
    states = {
        "category": torch.cat(category),
        "grasp_region": torch.cat(region),
        "grasp_side": torch.cat(side),
    }
    states["objective"] = torch.linspace(-1.0, 1.0, states["category"].numel())
    valid = torch.ones(states["category"].numel(), dtype=torch.bool)

    keep, trimmed = select_production_reset_rows(states, valid)

    assert int(keep.sum()) == 20_000
    assert int((keep & (states["category"] == 0)).sum()) == 10_000
    assert int((keep & (states["category"] == 1)).sum()) == 10_000
    assert int((keep & (states["grasp_region"] == 1)).sum()) == 1_000
    assert torch.equal(trimmed, valid & ~keep)

    near_side_zero = (states["grasp_region"] == 1) & (states["grasp_side"] == 0)
    valid[torch.nonzero(near_side_zero, as_tuple=False).flatten()[:3]] = False
    with pytest.raises(RuntimeError, match="near-pour grasping states for side 0"):
        select_production_reset_rows(states, valid)


def test_oriented_box_overlap_checks_all_axes():
    centers_a = torch.zeros((3, 3), dtype=torch.float64)
    centers_b = torch.tensor(((0.5, 0.0, 0.0), (2.1, 0.0, 0.0), (1.2, 1.2, 0.0)), dtype=torch.float64)
    identity = torch.zeros((3, 4), dtype=torch.float64)
    identity[:, 3] = 1.0
    rotated = identity.clone()
    rotated[2, 2] = math.sin(math.pi / 8.0)
    rotated[2, 3] = math.cos(math.pi / 8.0)

    overlap = oriented_boxes_overlap(
        centers_a,
        identity,
        (1.0, 1.0, 1.0),
        centers_b,
        rotated,
        (1.0, 1.0, 1.0),
    )

    assert torch.equal(overlap, torch.tensor((True, False, True)))


def test_near_pour_mask_requires_alignment_clearance_and_tilt():
    source_pose = torch.tensor(
        (
            (0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0),
            (0.2, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    target_pose = _identity_poses(torch.zeros((4, 3), dtype=torch.float64))

    valid = above_target_tilted_mask(
        source_pose,
        target_pose,
        cup_center_offset=(0.0, 0.0, 0.0),
        target_rim_height=0.05,
        max_horizontal_distance=0.02,
        min_vertical_clearance=0.15,
        min_tilt_angle=math.radians(120.0),
    )

    assert valid.tolist() == [True, False, False, False]


def test_oriented_box_support_requires_the_complete_footprint():
    poses = _identity_poses(torch.tensor(((0.5, 0.5, 0.0), (0.95, 0.5, 0.0)), dtype=torch.float64))

    supported = oriented_box_supported_by_bounds(
        poses,
        (0.1, 0.1, 0.1),
        (0.0, 0.0),
        (1.0, 1.0),
        clearance=0.01,
    )

    assert supported.tolist() == [True, False]


def test_reset_dataset_cache_payload_has_exact_schema_categories_values_and_hashes():
    payload = _tiny_payload()

    validate_reset_dataset(payload, expected_grasping_count=2, expected_non_grasping_count=2)
    assert payload["schema_version"] == 6
    assert payload["format"] == "franka_pour_reset_dataset"
    assert len(payload["contract_sha256"]) == 64
    assert len(payload["content_sha256"]) == 64
    assert torch.equal(payload["states"]["category"], torch.tensor((1, 1, 0, 0), dtype=torch.int8))
    torch.testing.assert_close(
        payload["states"]["objective"], torch.tensor((0.0, 1.0, -1.0, -1.0), dtype=torch.float32)
    )
    assert tuple(payload["particle_layouts"]["local_position"].shape) == (1, 3, 3)
    assert tuple(payload["particle_layouts"]["local_velocity"].shape) == (1, 3, 3)


def test_reset_dataset_cache_hashes_are_deterministic_and_detect_tensor_tampering():
    first = _tiny_payload()
    second = _tiny_payload()

    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["content_sha256"] == second["content_sha256"]

    tampered = deepcopy(first)
    tampered["states"]["arm_joint_position"][0, 0] += 0.125
    with pytest.raises(ValueError, match="(?i)content|hash"):
        validate_reset_dataset(tampered, expected_grasping_count=2, expected_non_grasping_count=2)


def test_reset_dataset_cache_round_trips_with_weights_only_loader(tmp_path):
    payload = _tiny_payload()
    path = tmp_path / "reset_dataset.pt"
    torch.save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)

    validate_reset_dataset(loaded, expected_grasping_count=2, expected_non_grasping_count=2)
    assert loaded["content_sha256"] == payload["content_sha256"]


def test_production_validation_rejects_candidate_dataset_without_dynamic_provenance():
    with pytest.raises(ValueError, match="dynamic_validation"):
        _validate_tiny_production_payload(_tiny_payload())


def test_production_validation_accepts_complete_dynamic_provenance():
    _validate_tiny_production_payload(_tiny_production_payload())


def test_production_validation_rejects_mismatched_current_task_contract():
    payload = _tiny_production_payload()
    current_contract = deepcopy(payload["metadata"]["task_contract"])
    current_contract["source_box_half"] = (0.030, 0.028, 0.0595)

    with pytest.raises(ValueError, match="task contract does not match"):
        validate_production_reset_dataset(
            payload,
            expected_grasping_count=2,
            expected_non_grasping_count=2,
            expected_task_contract=current_contract,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda marker: marker.update(source_content_sha256="A" * 64), "source_content_sha256"),
        (lambda marker: marker.update(steps=0), "steps"),
        (lambda marker: marker.update(settle_steps=60), "settle_steps"),
        (lambda marker: marker.update(failure_counts={}), "failure_counts"),
        (lambda marker: marker.update(balance_trimmed=-1), "balance_trimmed"),
    ),
)
def test_production_validation_rejects_malformed_dynamic_provenance(mutation, message):
    payload = _tiny_production_payload()
    mutation(payload["metadata"]["dynamic_validation"])
    payload["content_sha256"] = reset_dataset_content_sha256(payload)

    with pytest.raises(ValueError, match=message):
        _validate_tiny_production_payload(payload)


@pytest.mark.parametrize(
    ("field", "index", "value"),
    (
        ("category", 0, 2),
        ("objective", 0, 1.1),
        ("objective", 2, 0.0),
    ),
)
def test_reset_dataset_cache_rejects_invalid_category_or_objective_semantics(field, index, value):
    states = _tiny_states()
    states[field][index] = value

    with pytest.raises(ValueError):
        payload = _tiny_payload(states=states)
        validate_reset_dataset(payload, expected_grasping_count=2, expected_non_grasping_count=2)


def test_reset_dataset_cache_rejects_missing_required_state_and_wrong_expected_counts():
    states = _tiny_states()
    del states["source_root_pose"]
    with pytest.raises(ValueError):
        payload = _tiny_payload(states=states)
        validate_reset_dataset(payload, expected_grasping_count=2, expected_non_grasping_count=2)

    payload = _tiny_payload()
    with pytest.raises(ValueError):
        validate_reset_dataset(payload, expected_grasping_count=3, expected_non_grasping_count=2)


def test_reset_dataset_cache_rejects_grasp_target_without_required_close_command():
    states = _tiny_states()
    states["finger_joint_target"][0] = 0.028

    with pytest.raises(ValueError, match="reset target"):
        _tiny_payload(states=states)


def test_reset_dataset_cache_rejects_non_grasp_target_that_changes_sampled_opening():
    states = _tiny_states()
    states["finger_joint_target"][2, 0] -= 0.001

    with pytest.raises(ValueError, match="opening"):
        _tiny_payload(states=states)


def test_reset_dataset_task_id_selects_the_production_registration():
    spec = gym.spec(FRANKA_POUR_RESET_DATASET_TASK_ID)

    assert spec.kwargs["env_cfg_entry_point"].endswith(":FrankaPourEnvCfg_RESET_DATASET")
