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


def _run_backend_probe(backend: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(_PROBE), "--backend", backend],
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


@pytest.mark.isaacsim_ci
def test_franka_lift_physx_runtimes_preserve_grasp_and_clone_contracts() -> None:
    """Both PhysX runtimes should execute the same Lift action and isolated two-finger grasp contract."""
    if importlib.util.find_spec("isaacsim") is None:
        pytest.skip("Isaac Sim not installed")
    if importlib.util.find_spec("ovphysx") is None or importlib.util.find_spec("ovphysx.types") is None:
        pytest.skip("ovphysx wheel not installed")
    if not torch.cuda.is_available():
        pytest.skip("PhysX backend parity requires a CUDA device")

    results = {backend: _run_backend_probe(backend) for backend in ("isaacsim_physx", "ovphysx")}
    expected_joints = [f"panda_joint{i}" for i in range(1, 8)] + [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    for backend, result in results.items():
        assert result["asset_path"].endswith("/Robots/FrankaEmika/franka_panda.usda")
        assert result["asset_variants"] == {"Physics": "physx", "Colliders": "gripper_only"}
        assert result["object_spawner"] == "CuboidCfg"
        assert result["action_type"] == "RelativeJointPositionAction"
        assert result["action_joint_names"] == expected_joints
        assert result["reset_finger_position_mean"] == pytest.approx(0.026, abs=5.0e-4)
        assert result["reset_peak_force_max"] < 0.01
        assert result["dual_contact_fraction"] >= 0.75, f"{backend} did not establish dual-finger contact"
        assert 0 < result["contact_onset_step_max"] <= 8
        assert result["grasp_peak_force_min"] > 0.01
        assert result["isolated_peak_force_max"] < 0.01, f"{backend} leaked contact across clones"
        assert result["grasp_peak_arm_velocity_max"] < 2.0
        assert result["isolated_peak_arm_velocity_max"] < 2.0
        assert result["isolated_finger_position_mean"] > result["grasp_finger_position_mean"] + 5.0e-4
        assert result["mimic_error_max"] < 1.0e-3

    # Contacts need not integrate identically, but both runtimes must close to the same
    # mechanically constrained configuration within a policy-relevant position band.
    assert (
        abs(results["isaacsim_physx"]["grasp_finger_position_mean"] - results["ovphysx"]["grasp_finger_position_mean"])
        < 0.005
    )
    assert (
        abs(
            results["isaacsim_physx"]["grasp_peak_arm_velocity_max"] - results["ovphysx"]["grasp_peak_arm_velocity_max"]
        )
        < 0.1
    )
