# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for recording command configuration."""

import math
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend_flags", [["--physics", "isaacsim_physx"], ["--physics", "physx", "--xr"]])
def test_recording_defers_command_resampling(monkeypatch, tmp_path, backend_flags):
    """Recording defers command resampling beyond a realistic demonstration."""
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", "Isaac-Reach-Franka", *backend_flags, "--teleop_device", "spacemouse"])
    cfg, _, _ = record_demos.create_environment_config(str(tmp_path), "reach")

    resampling_time_range = cfg.commands.ee_pose.resampling_time_range
    assert all(math.isfinite(bound) for bound in resampling_time_range)
    assert min(resampling_time_range) > 24 * 60 * 60


@pytest.mark.parametrize("backend", ["newton", "ovphysx"])
def test_recording_rejects_kitless_before_device_or_recorder_setup(monkeypatch, tmp_path, backend):
    """Unsupported typed presets cannot reach a recorder path that skips task action policy."""
    from types import SimpleNamespace

    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_teleop.entrypoints import record_demos

    config = SimpleNamespace(
        sim=SimpleNamespace(physics=NewtonCfg() if backend == "newton" else OvPhysxCfg()), scene=SimpleNamespace()
    )
    monkeypatch.setattr(record_demos, "args_cli", SimpleNamespace(task="Test", device="cpu"), raising=False)
    monkeypatch.setattr(record_demos, "resolve_task_config", lambda *_args: (config, None))
    monkeypatch.setattr(record_demos, "select_teleop_cfg", lambda *_args: pytest.fail("device selection reached"))
    with pytest.raises(ValueError, match="currently supports Kit PhysX only"):
        record_demos.create_environment_config(str(tmp_path), "rejected")
