# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamic parity test for Franka Lift on Isaac Sim PhysX and OvPhysX."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_PROBE = Path(__file__).with_name("lift_physx_backend_probe.py")
_RESULT_PREFIX = "LIFT_PHYSX_BACKEND_PROBE="


def _run_backend_probe(backend: str, mode: str = "contact") -> dict:
    result = subprocess.run(
        [sys.executable, str(_PROBE), "--backend", backend, "--mode", mode],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{backend} Lift parity probe failed with exit code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    result_lines = [line for line in result.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    assert len(result_lines) == 1, f"Missing unique probe result for {backend}.\nstdout:\n{result.stdout}"
    return json.loads(result_lines[0][len(_RESULT_PREFIX) :])


def _skip_without_physx_runtimes() -> None:
    if importlib.util.find_spec("isaacsim") is None:
        pytest.skip("Isaac Sim not installed")
    if importlib.util.find_spec("ovphysx") is None or importlib.util.find_spec("ovphysx.types") is None:
        pytest.skip("ovphysx wheel not installed")
    if not torch.cuda.is_available():
        pytest.skip("PhysX backend parity requires a CUDA device")


@pytest.mark.isaacsim_ci
def test_franka_lift_physx_runtimes_preserve_grasp_and_clone_contracts() -> None:
    """Both PhysX runtimes should execute the same Lift action and isolated two-finger grasp contract."""
    _skip_without_physx_runtimes()

    results = {backend: _run_backend_probe(backend) for backend in ("isaacsim_physx", "ovphysx")}
    expected_joints = [f"panda_joint{i}" for i in range(1, 8)] + [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    for backend, result in results.items():
        arm_position_trajectory = torch.tensor(result["driven_arm_position_trajectory"])
        arm_velocity_trajectory = torch.tensor(result["driven_arm_velocity_trajectory"])
        assert result["asset_path"].endswith("/Robots/FrankaEmika/franka_panda.usda")
        assert result["asset_variants"] == {"Physics": "physx", "Colliders": "gripper_only"}
        assert result["object_spawner"] == "CuboidCfg"
        assert result["action_type"] == "RelativeJointPositionAction"
        assert result["action_joint_names"] == expected_joints
        assert result["enable_external_forces_every_iteration"] is True
        assert 0.0 < result["reset_finger_position_mean"] <= 0.04
        assert result["reset_peak_force_max"] < 0.01
        assert result["dual_contact_fraction"] >= 0.75, f"{backend} did not establish dual-finger contact"
        assert 0 < result["contact_onset_step_max"] <= 8
        assert result["grasp_peak_force_min"] > 0.01
        assert result["isolated_peak_force_max"] < 0.01, f"{backend} leaked contact across clones"
        assert result["grasp_peak_arm_velocity_max"] < 2.0
        assert result["isolated_peak_arm_velocity_max"] < 2.0
        assert arm_position_trajectory.shape == (32, 7)
        assert arm_velocity_trajectory.shape == (32, 7)
        assert arm_position_trajectory.abs().max() > 0.01
        assert arm_velocity_trajectory.abs().max() > 0.1
        assert result["driven_arm_position_clone_spread_max"] < 5.0e-4
        assert result["driven_arm_velocity_clone_spread_max"] < 1.5e-2
        assert result["isolated_finger_position_mean"] > result["grasp_finger_position_mean"] + 5.0e-4
        assert result["mimic_error_max"] < 1.0e-3

    # Contacts need not integrate identically, but both runtimes must close to the same
    # mechanically constrained configuration within a policy-relevant position band.
    assert (
        abs(results["isaacsim_physx"]["grasp_finger_position_mean"] - results["ovphysx"]["grasp_finger_position_mean"])
        < 0.005
    )
    assert results["ovphysx"]["reset_finger_position_mean"] == pytest.approx(
        results["isaacsim_physx"]["reset_finger_position_mean"], abs=1.0e-5
    )
    assert results["ovphysx"]["grasp_peak_force_min"] == pytest.approx(
        results["isaacsim_physx"]["grasp_peak_force_min"], rel=0.1, abs=0.5
    )
    assert (
        abs(
            results["isaacsim_physx"]["grasp_peak_arm_velocity_max"] - results["ovphysx"]["grasp_peak_arm_velocity_max"]
        )
        < 0.1
    )

    # Relative arm drives should produce the same policy-visible trajectory. Allow small
    # floating-point integration differences while still catching a changed drive contract.
    isaacsim_arm_initial = torch.tensor(results["isaacsim_physx"]["driven_arm_initial_position_mean"])
    ovphysx_arm_initial = torch.tensor(results["ovphysx"]["driven_arm_initial_position_mean"])
    torch.testing.assert_close(ovphysx_arm_initial, isaacsim_arm_initial, rtol=0.0, atol=1.0e-6)
    isaacsim_arm_position = torch.tensor(results["isaacsim_physx"]["driven_arm_position_trajectory"])
    ovphysx_arm_position = torch.tensor(results["ovphysx"]["driven_arm_position_trajectory"])
    torch.testing.assert_close(ovphysx_arm_position[:12], isaacsim_arm_position[:12], rtol=1.0e-2, atol=1.0e-4)
    isaacsim_arm_velocity = torch.tensor(results["isaacsim_physx"]["driven_arm_velocity_trajectory"])
    ovphysx_arm_velocity = torch.tensor(results["ovphysx"]["driven_arm_velocity_trajectory"])
    torch.testing.assert_close(ovphysx_arm_velocity[:12], isaacsim_arm_velocity[:12], rtol=1.0e-2, atol=1.0e-3)


@pytest.mark.isaacsim_ci
def test_franka_lift_physx_runtimes_match_randomized_training_trace() -> None:
    """The seeded cube training path should randomize and step identically on both PhysX runtimes."""
    _skip_without_physx_runtimes()

    results = {backend: _run_backend_probe(backend, "training") for backend in ("isaacsim_physx", "ovphysx")}
    isaacsim_result = results["isaacsim_physx"]
    ovphysx_result = results["ovphysx"]
    expected_properties = {
        "joint_stiffness",
        "joint_damping",
        "joint_friction",
        "joint_dynamic_friction",
        "joint_viscous_friction",
        "object_mass",
        "object_inertia",
        "robot_material",
        "object_material",
    }

    for result in results.values():
        assert result["mode"] == "training"
        assert result["asset_path"].endswith("/Robots/FrankaEmika/franka_panda.usda")
        assert result["asset_variants"] == {"Physics": "physx", "Colliders": "gripper_only"}
        assert result["object_spawner"] == "CuboidCfg"
        assert set(result["startup_properties"]) == expected_properties
        assert torch.tensor(result["startup_properties"]["joint_stiffness"]).std() > 0.0
        assert torch.tensor(result["startup_properties"]["joint_damping"]).std() > 0.0
        assert torch.tensor(result["startup_properties"]["object_mass"]).std() > 0.0
        assert torch.tensor(result["startup_properties"]["object_material"])[..., :2].std() > 0.0
        assert len(result["observation_trajectory"]) == 7
        assert not torch.tensor(result["terminated_trajectory"]).any()
        assert not torch.tensor(result["truncated_trajectory"]).any()
        assert not torch.tensor(result["termination_term_trajectory"]).any()

    for property_name in expected_properties:
        torch.testing.assert_close(
            torch.tensor(ovphysx_result["startup_properties"][property_name]),
            torch.tensor(isaacsim_result["startup_properties"][property_name]),
            rtol=0.0,
            atol=1.0e-6,
        )

    for state_name in ("reset_joint_position", "reset_object_pose", "command"):
        torch.testing.assert_close(
            torch.tensor(ovphysx_result[state_name]),
            torch.tensor(isaacsim_result[state_name]),
            rtol=0.0,
            atol=1.0e-6,
        )

    assert ovphysx_result["reward_terms"] == isaacsim_result["reward_terms"]
    assert ovphysx_result["termination_terms"] == isaacsim_result["termination_terms"]
    for group_name in ("policy", "proprio", "perception"):
        torch.testing.assert_close(
            torch.tensor([step[group_name] for step in ovphysx_result["observation_trajectory"]]),
            torch.tensor([step[group_name] for step in isaacsim_result["observation_trajectory"]]),
            rtol=1.0e-5,
            atol=2.0e-6,
        )
    torch.testing.assert_close(
        torch.tensor(ovphysx_result["reward_trajectory"]),
        torch.tensor(isaacsim_result["reward_trajectory"]),
        rtol=1.0e-5,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        torch.tensor(ovphysx_result["reward_term_trajectory"]),
        torch.tensor(isaacsim_result["reward_term_trajectory"]),
        rtol=1.0e-5,
        atol=1.0e-9,
    )
    assert ovphysx_result["terminated_trajectory"] == isaacsim_result["terminated_trajectory"]
    assert ovphysx_result["truncated_trajectory"] == isaacsim_result["truncated_trajectory"]
    assert ovphysx_result["termination_term_trajectory"] == isaacsim_result["termination_term_trajectory"]


@pytest.mark.isaacsim_ci
def test_franka_lift_physx_runtimes_match_complete_random_policy_rollouts() -> None:
    """Complete randomized episodes should preserve aggregate contact and reset behavior across runtimes."""
    _skip_without_physx_runtimes()

    results = {backend: _run_backend_probe(backend, "rollout") for backend in ("isaacsim_physx", "ovphysx")}
    isaacsim_result = results["isaacsim_physx"]
    ovphysx_result = results["ovphysx"]
    for result in results.values():
        assert result["mode"] == "rollout"
        assert result["asset_variants"] == {"Physics": "physx", "Colliders": "gripper_only"}
        assert result["num_envs"] == 256
        assert result["num_steps"] == 720
        assert result["nonfinite_state_count"] == 0
        assert result["dual_contact_fraction"] > 0.0
        assert sum(result["termination_counts"]) > result["num_envs"]

    assert ovphysx_result["reward_terms"] == isaacsim_result["reward_terms"]
    assert ovphysx_result["termination_terms"] == isaacsim_result["termination_terms"]
    assert ovphysx_result["mean_return"] == pytest.approx(isaacsim_result["mean_return"], abs=0.1)
    torch.testing.assert_close(
        torch.tensor(ovphysx_result["mean_reward_terms"]),
        torch.tensor(isaacsim_result["mean_reward_terms"]),
        rtol=0.0,
        atol=5.0e-4,
    )
    torch.testing.assert_close(
        torch.tensor(ovphysx_result["termination_counts"]),
        torch.tensor(isaacsim_result["termination_counts"]),
        rtol=0.0,
        atol=24.0,
    )
    assert ovphysx_result["dual_contact_fraction"] == pytest.approx(
        isaacsim_result["dual_contact_fraction"], abs=5.0e-5
    )
    assert ovphysx_result["object_above_table_fraction"] == pytest.approx(
        isaacsim_result["object_above_table_fraction"], abs=1.0e-3
    )
    assert ovphysx_result["mean_finger_separation"] == pytest.approx(
        isaacsim_result["mean_finger_separation"], abs=1.0e-3
    )
    assert ovphysx_result["max_joint_velocity"] == pytest.approx(isaacsim_result["max_joint_velocity"], abs=2.0)
