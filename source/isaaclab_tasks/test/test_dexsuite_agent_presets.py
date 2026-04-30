# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Dexsuite env/agent preset coupling."""

import sys
from unittest import mock

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.camera_cfg import (
    ResNetSingleCameraObservationsCfg,
)
from isaaclab_tasks.utils import resolve_task_config


def _resolve_lift_with_presets(presets: str):
    with mock.patch.object(sys, "argv", [sys.argv[0], f"presets={presets}"]):
        return resolve_task_config("Isaac-Dexsuite-Kuka-Allegro-Lift-v0", "rsl_rl_cfg_entry_point")


def test_resnet_single_camera_preset_selects_matching_agent():
    """ResNet camera preset should switch both env observations and the default RSL-RL agent."""
    env_cfg, agent_cfg = _resolve_lift_with_presets("cube,resnet_single_camera,isaacsim_rtx_renderer")

    assert isinstance(env_cfg.observations, ResNetSingleCameraObservationsCfg)
    assert env_cfg.scene.base_camera is not None
    assert agent_cfg.experiment_name == "dexsuite_kuka_allegro_resnet_features"
    assert agent_cfg.obs_groups == {
        "actor": ["policy", "proprio", "resnet_features"],
        "critic": ["policy", "proprio", "resnet_features"],
    }
