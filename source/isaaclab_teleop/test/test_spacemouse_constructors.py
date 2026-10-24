# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the SpaceMouse ``IsaacTeleopCfg`` builders."""

from __future__ import annotations

import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop.spacemouse.se2_spacemouse import se2_spacemouse_teleop_cfg
from isaaclab_teleop.spacemouse.se3_spacemouse import se3_spacemouse_teleop_cfg


def _has_spacemouse_source(cfg) -> bool:
    """Whether the config's pipeline reads the SpaceMouse."""
    from isaacteleop.retargeting_engine.deviceio_source_nodes import SpaceMouseSource, find_sources

    return len(find_sources(cfg.pipeline_builder(), SpaceMouseSource)) == 1


def _starts_the_spacemouse_plugin(cfg) -> bool:
    """Whether the config starts the Isaac Capture spacemouse plugin that feeds its source."""
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR

    return [(p.plugin_name, p.plugin_root_id, p.search_paths) for p in cfg.plugins] == [
        ("spacemouse", "spacemouse", [BUNDLED_PLUGINS_DIR])
    ]


# ---------------------------------------------------------------------------
# se2_spacemouse_teleop_cfg / se3_spacemouse_teleop_cfg
# ---------------------------------------------------------------------------


class TestSe3SpaceMouseTeleopCfg:
    def test_defaults(self):
        cfg = se3_spacemouse_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabSpaceMouseSe3"
        assert _starts_the_spacemouse_plugin(cfg)
        assert _has_spacemouse_source(cfg)
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se3_spacemouse_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"


class TestSe2SpaceMouseTeleopCfg:
    def test_defaults(self):
        cfg = se2_spacemouse_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabSpaceMouseSe2"
        assert _starts_the_spacemouse_plugin(cfg)
        assert _has_spacemouse_source(cfg)
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_spacemouse_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"
