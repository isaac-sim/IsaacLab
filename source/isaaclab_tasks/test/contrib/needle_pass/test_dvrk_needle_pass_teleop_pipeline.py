# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused integration tests for the dVRK needle-pass IsaacTeleop graph.

The external dVRK nodes are supplied by NVIDIA/IsaacTeleop PR 769.  Generic
task shards skip this module until that API is released; the focused PR lane
installs the immutable test pin before running this file.
"""

import numpy as np
import pytest

pytest.importorskip("isaacteleop.retargeters.DVRK")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

from isaacteleop.retargeters import DVRKPSMClutchRetargeter, DVRKPSMGripperRetargeter  # noqa: E402
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource  # noqa: E402

from isaaclab_tasks.contrib.needle_pass.config.dvrk.ik_abs_env_cfg import (  # noqa: E402
    _TELEOP_AVAILABLE,
    DONOR_GRASP_CLOSEDNESS,
    DONOR_GRASP_JAW_POS,
    DVRK_PSM_JAW_CLOSED_POS,
    LEFT_TOOL_HOME_POS_W,
    LEFT_TOOL_HOME_ROT_XYZW,
    LEFT_WORKSPACE_LOWER,
    LEFT_WORKSPACE_UPPER,
    RIGHT_TOOL_HOME_POS_W,
    RIGHT_TOOL_HOME_ROT_XYZW,
    RIGHT_WORKSPACE_LOWER,
    RIGHT_WORKSPACE_UPPER,
    _build_dvrk_needle_pass_pipeline,
)

_EXPECTED_ACTION_ORDER = [
    "left_pos_x",
    "left_pos_y",
    "left_pos_z",
    "left_quat_x",
    "left_quat_y",
    "left_quat_z",
    "left_quat_w",
    "left_jaw_1",
    "left_jaw_2",
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_quat_x",
    "right_quat_y",
    "right_quat_z",
    "right_quat_w",
    "right_jaw_1",
    "right_jaw_2",
]


def _reorderer_subgraph():
    pipeline = _build_dvrk_needle_pass_pipeline()
    return pipeline, pipeline.output_mapping["action"].module


def test_dvrk_pipeline_action_is_18d():
    """The pipeline emits the task's ``7 + 2 + 7 + 2`` action ABI."""
    pipeline = _build_dvrk_needle_pass_pipeline()

    assert _TELEOP_AVAILABLE
    assert pipeline.output_types()["action"].types[0].shape == (18,)


def test_dvrk_pipeline_output_order_matches_action_terms():
    """The flattened values resolve in left pose/jaws then right pose/jaws order."""
    _, subgraph = _reorderer_subgraph()
    try:
        output_order = subgraph._target_module._output_order
    except AttributeError:
        pytest.skip("IsaacTeleop does not expose graph wiring for order inspection")

    assert output_order == _EXPECTED_ACTION_ORDER


def test_dvrk_pipeline_routes_world_transformed_controller_sides():
    """Each PSM consumes its own controller after the shared world transform."""
    _, reorderer_subgraph = _reorderer_subgraph()
    try:
        connections = reorderer_subgraph._input_connections
        left_pose = connections["left_pose"].module
        left_jaws = connections["left_jaws"].module
        right_pose = connections["right_pose"].module
        right_jaws = connections["right_jaws"].module
    except AttributeError:
        pytest.skip("IsaacTeleop does not expose graph wiring for route inspection")

    assert isinstance(left_pose._target_module, DVRKPSMClutchRetargeter)
    assert isinstance(left_jaws._target_module, DVRKPSMGripperRetargeter)
    assert isinstance(right_pose._target_module, DVRKPSMClutchRetargeter)
    assert isinstance(right_jaws._target_module, DVRKPSMGripperRetargeter)
    assert list(left_pose._target_module.input_spec()) == [ControllersSource.LEFT]
    assert list(left_jaws._target_module.input_spec()) == [ControllersSource.LEFT]
    assert list(right_pose._target_module.input_spec()) == [ControllersSource.RIGHT]
    assert list(right_jaws._target_module.input_spec()) == [ControllersSource.RIGHT]

    left_transform = left_pose._input_connections[ControllersSource.LEFT].module
    right_transform = right_pose._input_connections[ControllersSource.RIGHT].module
    assert left_jaws._input_connections[ControllersSource.LEFT].module is left_transform
    assert right_jaws._input_connections[ControllersSource.RIGHT].module is right_transform
    assert left_transform is right_transform
    assert left_transform._input_connections["transform"].module.name == "world_T_anchor"


def test_dvrk_pipeline_preserves_side_homes_workspaces_and_reset_jaws():
    """Each side retains its calibrated world limits and intended reset jaw state."""
    _, reorderer_subgraph = _reorderer_subgraph()
    try:
        connections = reorderer_subgraph._input_connections
        left_clutch_cfg = connections["left_pose"].module._target_module._clutch_state._config
        right_clutch_cfg = connections["right_pose"].module._target_module._clutch_state._config
        left_gripper_cfg = connections["left_jaws"].module._target_module._jaw_intent._config
        right_gripper_cfg = connections["right_jaws"].module._target_module._jaw_intent._config
    except AttributeError:
        pytest.skip("IsaacTeleop does not expose graph node configs for contract inspection")

    np.testing.assert_allclose(left_clutch_cfg.home_position, LEFT_TOOL_HOME_POS_W)
    np.testing.assert_allclose(left_clutch_cfg.home_orientation, LEFT_TOOL_HOME_ROT_XYZW)
    assert left_clutch_cfg.workspace_lower == LEFT_WORKSPACE_LOWER
    assert left_clutch_cfg.workspace_upper == LEFT_WORKSPACE_UPPER
    np.testing.assert_allclose(right_clutch_cfg.home_position, RIGHT_TOOL_HOME_POS_W)
    np.testing.assert_allclose(right_clutch_cfg.home_orientation, RIGHT_TOOL_HOME_ROT_XYZW)
    assert right_clutch_cfg.workspace_lower == RIGHT_WORKSPACE_LOWER
    assert right_clutch_cfg.workspace_upper == RIGHT_WORKSPACE_UPPER

    assert DONOR_GRASP_CLOSEDNESS == 1.0
    assert DONOR_GRASP_JAW_POS == DVRK_PSM_JAW_CLOSED_POS
    assert left_gripper_cfg.initial_closedness == DONOR_GRASP_CLOSEDNESS
    assert left_gripper_cfg.jaw_closed == DONOR_GRASP_JAW_POS
    assert right_gripper_cfg.initial_closedness == 0.0
