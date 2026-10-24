# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the gamepad :class:`~isaaclab_teleop.IsaacTeleopCfg` builders."""

from __future__ import annotations

import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop.gamepad.se2_gamepad import se2_gamepad_teleop_cfg
from isaaclab_teleop.gamepad.se3_gamepad import se3_gamepad_teleop_cfg


def _has_gamepad_source(cfg) -> bool:
    """Whether the config's pipeline reads the gamepad."""
    from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource, find_sources

    return len(find_sources(cfg.pipeline_builder(), GamepadSource)) == 1


def _starts_the_gamepad_plugin(cfg) -> bool:
    """Whether the config starts the Isaac Capture gamepad plugin that feeds its source."""
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR

    return [(p.plugin_name, p.plugin_root_id, p.search_paths) for p in cfg.plugins] == [
        ("gamepad", "gamepad", [BUNDLED_PLUGINS_DIR])
    ]


class TestSe3GamepadTeleopCfg:
    def test_defaults(self):
        cfg = se3_gamepad_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabGamepadSe3"
        assert _starts_the_gamepad_plugin(cfg)
        assert _has_gamepad_source(cfg)
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se3_gamepad_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"


class TestSe2GamepadTeleopCfg:
    def test_defaults(self):
        cfg = se2_gamepad_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabGamepadSe2"
        assert _starts_the_gamepad_plugin(cfg)
        assert _has_gamepad_source(cfg)
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_gamepad_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"
