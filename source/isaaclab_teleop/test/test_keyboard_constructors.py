# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the keyboard ``IsaacTeleopCfg`` builders."""

from __future__ import annotations

import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop.keyboard.se2_keyboard import se2_keyboard_teleop_cfg
from isaaclab_teleop.keyboard.se3_keyboard import se3_keyboard_teleop_cfg

# ---------------------------------------------------------------------------
# se2_keyboard_teleop_cfg / se3_keyboard_teleop_cfg
# ---------------------------------------------------------------------------


def _keyboard_sources(pipeline):
    from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource, find_sources

    return find_sources(pipeline, KeyboardSource)


class TestSe3KeyboardTeleopCfg:
    def test_defaults(self):
        cfg = se3_keyboard_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabKeyboardSe3"
        # the keyboard runs in process: no plugin
        assert cfg.plugins == []
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se3_keyboard_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"

    def test_pipeline_reads_one_keyboard(self):
        assert len(_keyboard_sources(se3_keyboard_teleop_cfg().pipeline_builder())) == 1

    @pytest.mark.parametrize(("gripper_term", "action_size"), [(True, 7), (False, 6)])
    def test_action_is_the_pose_delta_plus_an_optional_gripper(self, gripper_term, action_size):
        """The action is ``[dx, dy, dz, drx, dry, drz]``, plus the gripper command when ``gripper_term``."""
        pipeline = se3_keyboard_teleop_cfg(gripper_term=gripper_term).pipeline_builder()

        (action,) = pipeline.output_types()["action"].types
        assert action.shape == (action_size,)


class TestSe2KeyboardTeleopCfg:
    def test_defaults(self):
        cfg = se2_keyboard_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabKeyboardSe2"
        assert cfg.plugins == []
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_keyboard_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"

    def test_pipeline_reads_one_keyboard(self):
        assert len(_keyboard_sources(se2_keyboard_teleop_cfg().pipeline_builder())) == 1
