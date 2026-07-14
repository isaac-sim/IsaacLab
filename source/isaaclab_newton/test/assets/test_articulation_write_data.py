# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Tests for Newton articulation simulation writes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from isaaclab_newton.assets.articulation.articulation import Articulation


@pytest.fixture
def articulation() -> Articulation:
    """Create an articulation shell with mocked simulation bindings."""
    articulation = object.__new__(Articulation)
    articulation._instantaneous_wrench_composer = MagicMock()
    articulation._permanent_wrench_composer = MagicMock()
    articulation._data = SimpleNamespace(
        _joint_pos_target=MagicMock(),
        _joint_vel_target=MagicMock(),
        _joint_effort_target=MagicMock(),
        _sim_bind_joint_position_target=MagicMock(),
        _sim_bind_joint_velocity_target=MagicMock(),
        _sim_bind_joint_act=MagicMock(),
        _sim_bind_joint_effort=MagicMock(),
        _sim_bind_body_external_wrench=MagicMock(),
    )
    articulation._has_newton_actuators = True
    articulation._device = "cpu"
    articulation._root_view = SimpleNamespace(count=2, link_count=3)
    articulation._ALL_ENV_MASK = MagicMock()
    articulation._ALL_BODY_MASK = MagicMock()
    articulation._initialize_handle = None
    articulation._invalidate_initialize_handle = None
    articulation._prim_deletion_handle = None
    articulation._debug_vis_handle = None
    articulation._physics_ready_handle = None
    return articulation


def test_write_data_to_sim_skips_inactive_instantaneous_wrench_reset(articulation: Articulation):
    """Test that an inactive instantaneous wrench composer is not reset."""
    articulation._instantaneous_wrench_composer.active = False
    articulation._permanent_wrench_composer.active = False

    articulation.write_data_to_sim()

    articulation._instantaneous_wrench_composer.reset.assert_not_called()


@patch("isaaclab_newton.assets.articulation.articulation.wp.launch")
def test_write_data_to_sim_clears_active_instantaneous_wrenches(launch: MagicMock, articulation: Articulation):
    """Test that active instantaneous wrenches are applied once and then cleared."""
    articulation._instantaneous_wrench_composer.active = True
    articulation._permanent_wrench_composer.active = True

    articulation.write_data_to_sim()

    launch.assert_called_once()
    articulation._instantaneous_wrench_composer.add_raw_buffers_from.assert_called_once_with(
        articulation._permanent_wrench_composer
    )
    articulation._instantaneous_wrench_composer.compose_to_body_frame.assert_called_once_with()
    articulation._instantaneous_wrench_composer.reset.assert_called_once_with()
    articulation._permanent_wrench_composer.reset.assert_not_called()


@patch("isaaclab_newton.assets.articulation.articulation.wp.launch")
def test_write_data_to_sim_preserves_permanent_wrenches(launch: MagicMock, articulation: Articulation):
    """Test that permanent wrenches are applied without resetting either composer."""
    articulation._instantaneous_wrench_composer.active = False
    articulation._permanent_wrench_composer.active = True

    articulation.write_data_to_sim()

    launch.assert_called_once()
    articulation._permanent_wrench_composer.compose_to_body_frame.assert_called_once_with()
    articulation._instantaneous_wrench_composer.reset.assert_not_called()
    articulation._permanent_wrench_composer.reset.assert_not_called()
