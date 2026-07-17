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

from isaacteleop.retargeting_engine.interface import (  # noqa: E402
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    GraphTime,
    OutputCombiner,
    TensorGroup,
)
from isaacteleop.schema import (  # noqa: E402
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    ControllerSnapshotTrackedT,
    Point,
    Pose,
    Quaternion,
)

from isaaclab_tasks.contrib.needle_pass.config.dvrk.ik_abs_env_cfg import (  # noqa: E402
    _TELEOP_AVAILABLE,
    DONOR_GRASP_JAW_POS,
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

from isaaclab_assets.robots.dvrk import DVRK_PSM_JAW_CLOSED_POS, DVRK_PSM_JAW_OPEN_POS  # noqa: E402


def _tracked_controller(
    position: tuple[float, float, float],
    *,
    squeeze: float = 1.0,
    trigger: float = 0.5,
) -> ControllerSnapshotTrackedT:
    """Build one valid tracked controller sample for public graph execution."""

    pose = Pose(Point(*position), Quaternion(0.0, 0.0, 0.0, 1.0))
    controller_pose = ControllerPose(pose, True)
    inputs = ControllerInputState(
        primary_click=False,
        secondary_click=False,
        thumbstick_click=False,
        menu_click=False,
        thumbstick_x=0.0,
        thumbstick_y=0.0,
        squeeze_value=squeeze,
        trigger_value=trigger,
    )
    return ControllerSnapshotTrackedT(ControllerSnapshot(controller_pose, controller_pose, inputs))


def _pipeline_inputs(
    pipeline: OutputCombiner,
    left_controller: ControllerSnapshotTrackedT,
    right_controller: ControllerSnapshotTrackedT,
) -> dict:
    """Build leaf-keyed inputs with an identity anchor-to-world transform."""

    leaf_nodes = {node.name: node for node in pipeline.get_leaf_nodes()}
    assert set(leaf_nodes) == {"controllers", "world_T_anchor"}

    controller_spec = leaf_nodes["controllers"].input_spec()
    left_group = TensorGroup(controller_spec["deviceio_controller_left"])
    left_group[0] = left_controller
    right_group = TensorGroup(controller_spec["deviceio_controller_right"])
    right_group[0] = right_controller

    transform_group = TensorGroup(leaf_nodes["world_T_anchor"].input_spec()["value"])
    transform_group[0] = np.eye(4, dtype=np.float32)
    return {
        "controllers": {
            "deviceio_controller_left": left_group,
            "deviceio_controller_right": right_group,
        },
        "world_T_anchor": {"value": transform_group},
    }


def _running_context(time_ns: int, *, reset: bool = False) -> ComputeContext:
    """Build a deterministic running-session context for one graph step."""

    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=time_ns, real_time_ns=time_ns),
        execution_events=ExecutionEvents(reset=reset, execution_state=ExecutionState.RUNNING),
    )


def _expected_initial_action() -> np.ndarray:
    """Return the task-ordered left pose/jaws then right pose/jaws reset action."""

    return np.asarray(
        LEFT_TOOL_HOME_POS_W
        + LEFT_TOOL_HOME_ROT_XYZW
        + DONOR_GRASP_JAW_POS
        + RIGHT_TOOL_HOME_POS_W
        + RIGHT_TOOL_HOME_ROT_XYZW
        + DVRK_PSM_JAW_OPEN_POS,
        dtype=np.float32,
    )


def test_dvrk_pipeline_action_is_18d():
    """The pipeline emits the task's ``7 + 2 + 7 + 2`` action ABI."""
    pipeline = _build_dvrk_needle_pass_pipeline()

    assert _TELEOP_AVAILABLE
    assert pipeline.output_types()["action"].types[0].shape == (18,)


def test_dvrk_pipeline_executes_tracked_controller_samples():
    """Public execution emits ordered homes, independent side motion, and tracking-loss holds."""

    pipeline = _build_dvrk_needle_pass_pipeline()
    assert isinstance(pipeline, OutputCombiner)
    initial_left = _tracked_controller((0.10, 0.20, 0.30))
    initial_right = _tracked_controller((-0.10, -0.20, -0.30))

    initial_outputs = pipeline.execute_pipeline(
        _pipeline_inputs(pipeline, initial_left, initial_right),
        _running_context(0, reset=True),
    )
    initial_action = initial_outputs["action"][0]
    expected_initial_action = _expected_initial_action()
    assert initial_action.shape == (18,)
    np.testing.assert_allclose(initial_action, expected_initial_action, atol=1.0e-6, rtol=0.0)

    left_delta = np.asarray((0.01, -0.02, 0.03), dtype=np.float32)
    moved_left = _tracked_controller(tuple(np.asarray((0.10, 0.20, 0.30)) + left_delta))
    moved_outputs = pipeline.execute_pipeline(
        _pipeline_inputs(pipeline, moved_left, initial_right),
        _running_context(1_000_000_000),
    )
    moved_action = moved_outputs["action"][0]
    expected_moved_action = expected_initial_action.copy()
    expected_moved_action[:3] += left_delta
    np.testing.assert_allclose(moved_action, expected_moved_action, atol=1.0e-6, rtol=0.0)

    right_delta = np.asarray((-0.02, 0.01, 0.04), dtype=np.float32)
    moved_right = _tracked_controller(tuple(np.asarray((-0.10, -0.20, -0.30)) + right_delta))
    tracking_loss_outputs = pipeline.execute_pipeline(
        _pipeline_inputs(pipeline, ControllerSnapshotTrackedT(), moved_right),
        _running_context(2_000_000_000),
    )
    tracking_loss_action = tracking_loss_outputs["action"][0]
    expected_tracking_loss_action = expected_moved_action.copy()
    expected_tracking_loss_action[9:12] += right_delta
    np.testing.assert_allclose(tracking_loss_action, expected_tracking_loss_action, atol=1.0e-6, rtol=0.0)


def test_dvrk_pipeline_clips_each_side_to_its_world_workspace():
    """Public execution applies each side's configured world-frame bounds."""

    pipeline = _build_dvrk_needle_pass_pipeline()
    initial_left_position = np.asarray((0.10, 0.20, 0.30))
    initial_right_position = np.asarray((-0.10, -0.20, -0.30))
    initial_left = _tracked_controller(tuple(initial_left_position))
    initial_right = _tracked_controller(tuple(initial_right_position))
    pipeline.execute_pipeline(
        _pipeline_inputs(pipeline, initial_left, initial_right),
        _running_context(0, reset=True),
    )

    extreme_left = _tracked_controller(tuple(initial_left_position + np.asarray((1.0, -1.0, 1.0))))
    extreme_right = _tracked_controller(tuple(initial_right_position + np.asarray((-1.0, 1.0, -1.0))))
    clipped_outputs = pipeline.execute_pipeline(
        _pipeline_inputs(pipeline, extreme_left, extreme_right),
        _running_context(1_000_000_000),
    )
    expected_action = _expected_initial_action()
    expected_action[:3] = (LEFT_WORKSPACE_UPPER[0], LEFT_WORKSPACE_LOWER[1], LEFT_WORKSPACE_UPPER[2])
    expected_action[9:12] = (RIGHT_WORKSPACE_LOWER[0], RIGHT_WORKSPACE_UPPER[1], RIGHT_WORKSPACE_LOWER[2])
    np.testing.assert_allclose(clipped_outputs["action"][0], expected_action, atol=1.0e-6, rtol=0.0)


def test_dvrk_pipeline_maps_independent_trigger_intent_to_ordered_jaws():
    """Public execution maps trigger changes to the correct two-jaw action slices."""

    pipeline = _build_dvrk_needle_pass_pipeline()
    left_position = (0.10, 0.20, 0.30)
    right_position = (-0.10, -0.20, -0.30)
    pipeline.execute_pipeline(
        _pipeline_inputs(
            pipeline,
            _tracked_controller(left_position, trigger=0.5),
            _tracked_controller(right_position, trigger=0.5),
        ),
        _running_context(0, reset=True),
    )

    output = pipeline.execute_pipeline(
        _pipeline_inputs(
            pipeline,
            _tracked_controller(left_position, trigger=0.5),
            _tracked_controller(right_position, trigger=0.8),
        ),
        _running_context(100_000_000),
    )["action"][0]
    expected_action = _expected_initial_action()
    expected_action[16:18] = np.asarray(DVRK_PSM_JAW_OPEN_POS) + 0.3 * (
        np.asarray(DVRK_PSM_JAW_CLOSED_POS) - np.asarray(DVRK_PSM_JAW_OPEN_POS)
    )
    np.testing.assert_allclose(output, expected_action, atol=1.0e-6, rtol=0.0)
