# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the SO-101 Capture graph with encoded controller snapshots and verify its 8D action contract."""

import numpy as np
import pytest

pytest.importorskip("isaacteleop")

from isaacteleop.retargeting_engine.interface import ComputeContext, ExecutionEvents, ExecutionState, TensorGroup
from isaacteleop.schema import ControllerInputState, ControllerPose, ControllerSnapshot, Point, Pose, Quaternion

from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import _build_so101_stack_pipeline


def test_so101_pipeline_tracks_engage_relative_pose_and_gripper():
    """Pose/closedness occupy the declared slots, and clutching holds the pose across release and re-engage."""
    pipeline = _build_so101_stack_pipeline()
    assert pipeline.output_types()["action"].types[0].shape == (8,)
    inputs = {
        leaf.name: {name: TensorGroup(group_type) for name, group_type in leaf.input_spec().items()}
        for leaf in pipeline.get_leaf_nodes()
    }
    inputs["world_T_anchor"]["value"][0] = np.eye(4, dtype=np.float32)
    inputs["controllers"]["deviceio_controller_left"][0] = None

    def step(position, orientation, squeeze=1.0, trigger=1.0, reset=False):
        inputs["controllers"]["deviceio_controller_right"][0] = ControllerSnapshot(
            grip_pose=ControllerPose(Pose(Point(*position), Quaternion(*orientation)), True),
            inputs=ControllerInputState(False, False, False, False, 0.0, 0.0, squeeze, trigger),
        )
        context = ComputeContext(execution_events=ExecutionEvents(reset=reset, execution_state=ExecutionState.RUNNING))
        action = pipeline.execute_pipeline(inputs, context)["action"][0]
        assert action.shape == (8,) and action.dtype == np.float32
        return action.copy()

    home = [0.01918, -0.18852, 0.18887, 0.0, 0.0, 0.0, 1.0, 1.0]
    # The first squeeze emits the configured home even with a rotated controller.
    np.testing.assert_allclose(step((0.2, 0.3, 0.4), (0.0, 0.0, 2**-0.5, 2**-0.5)), home, atol=1e-6)
    # Relative to the initial Z quarter-turn, this controller quaternion gives [1, 2, 3, 4] / sqrt(30).
    moved = [0.11918, 0.01148, 0.48887, *np.array([1, 2, 3, 4]) / np.sqrt(30), 1.0]
    np.testing.assert_allclose(step((0.3, 0.5, 0.7), np.array([3, 1, 7, 1]) / np.sqrt(60)), moved, atol=1e-6)
    # Release at another pose, then re-engage there: the commanded arm pose stays held.
    held = [*moved[:7], 0.0]
    np.testing.assert_allclose(step((0.8, 0.9, 1.0), (0.0, 0.0, 0.0, 1.0), squeeze=0.0, trigger=0.0), held, atol=1e-6)
    np.testing.assert_allclose(step((0.8, 0.9, 1.0), (0.0, 0.0, 0.0, 1.0), trigger=0.0), held, atol=1e-6)
    np.testing.assert_allclose(step((0.8, 0.9, 1.0), (0.0, 0.0, 0.0, 1.0), reset=True), home, atol=1e-6)
