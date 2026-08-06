# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deterministic physics configuration in ``isaaclab.app.sim_launcher``."""

import argparse

from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.app import scan


def test_deterministic_flag_selects_newton_gpu_to_gpu_mode() -> None:
    """Test that ``--deterministic`` selects Newton's strongest deterministic mode."""
    physics_cfg = NewtonCfg(deterministic_mode="run_to_run")

    config_scan = scan(physics_cfg, argparse.Namespace(deterministic=True))

    assert config_scan.effective_cfg is physics_cfg
    assert physics_cfg.deterministic_mode == "gpu_to_gpu"


def test_deterministic_flag_applies_after_physics_override() -> None:
    """Test that a CLI-selected Newton backend also receives deterministic settings."""
    config_scan = scan(PhysxCfg(), {"physics": "newton_mjwarp", "deterministic": True})

    assert isinstance(config_scan.effective_cfg, NewtonCfg)
    assert config_scan.effective_cfg.deterministic_mode == "gpu_to_gpu"


def test_omitted_deterministic_flag_preserves_newton_mode() -> None:
    """Test that programmatic Newton determinism remains unchanged without the flag."""
    physics_cfg = NewtonCfg(deterministic_mode="run_to_run")

    scan(physics_cfg)

    assert physics_cfg.deterministic_mode == "run_to_run"
    assert physics_cfg.to_dict()["deterministic_mode"] == "run_to_run"
