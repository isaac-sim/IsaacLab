# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the real SO-101 Capture graph after hardware controller conversion, without XR/GPU."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("isaacteleop")

from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaacteleop.retargeting_engine.interface.retargeter_core_types import ExecutionCache, GraphTime
from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex, TransformMatrix

from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import (
    _BASE_T_GRIPPER_HOME,
    _build_so101_stack_pipeline,
)


def test_so101_pipeline_full_pose_clutch_and_action_order():
    """Home, motion, re-clutch and reset survive the real transform/retarget/flatten graph."""
    pipeline = _build_so101_stack_pipeline()
    controllers = next(node for node in pipeline.get_leaf_nodes() if isinstance(node, ControllersSource))
    anchor = TensorGroup(TransformMatrix())
    anchor[0] = np.eye(4, dtype=np.float32)

    def step(position, rotation, *, squeeze=1.0, trigger=0.0, reset=False):
        controller = TensorGroup(ControllerInput())
        # Populate the same complete typed input as ControllersSource, replacing hardware polling
        # only. Cached source output still traverses the real frame transform and all retargeters.
        for index in ControllerInputIndex:
            if index in (ControllerInputIndex.GRIP_POSITION, ControllerInputIndex.AIM_POSITION):
                controller[index] = np.asarray(position, dtype=np.float32)
            elif index in (ControllerInputIndex.GRIP_ORIENTATION, ControllerInputIndex.AIM_ORIENTATION):
                controller[index] = rotation.as_quat().astype(np.float32)
            elif index in (ControllerInputIndex.GRIP_IS_VALID, ControllerInputIndex.AIM_IS_VALID):
                controller[index] = True
            else:
                controller[index] = 0.0
        controller[ControllerInputIndex.SQUEEZE_VALUE] = squeeze
        controller[ControllerInputIndex.TRIGGER_VALUE] = trigger
        context = ComputeContext(
            graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
            execution_events=ExecutionEvents(reset=reset, execution_state=ExecutionState.RUNNING),
        )
        cache = ExecutionCache({"world_T_anchor": {"value": anchor}}, context)
        cache.cache(
            id(controllers),
            {ControllersSource.LEFT: OptionalTensorGroup(ControllerInput()), ControllersSource.RIGHT: controller},
        )
        action = np.from_dlpack(pipeline.execute_pipeline_with_cache(cache)["action"][0]).copy()
        assert action.shape == (8,)
        assert np.isfinite(action).all()
        return action

    def check(action, position, rotation, gripper):
        np.testing.assert_allclose(action[:3], position, atol=1e-6)
        np.testing.assert_allclose(Rotation.from_quat(action[3:7]).as_matrix(), rotation.as_matrix(), atol=1e-6)
        assert action[7] == pytest.approx(gripper)

    home_position = _BASE_T_GRIPPER_HOME[:3, 3]
    home_rotation = Rotation.from_matrix(_BASE_T_GRIPPER_HOME[:3, :3])
    origin = Rotation.from_euler("xyz", [0.2, -0.4, 0.3])
    position = np.array([0.3, -0.1, 0.5])
    check(step(position, origin, squeeze=0.0, reset=True), home_position, home_rotation, 0.0)
    check(step(position, origin), home_position, home_rotation, 0.0)
    delta = Rotation.from_rotvec([0.0, 0.0, 0.3])
    displacement = np.array([0.02, -0.03, 0.01])
    moved_position = home_position + displacement
    moved_rotation = delta * home_rotation
    check(step(position + displacement, delta * origin, trigger=1.0), moved_position, moved_rotation, 1.0)
    other = Rotation.from_euler("xyz", [-0.5, 0.2, 0.8])
    check(step(-position, other, squeeze=0.0), moved_position, moved_rotation, 0.0)
    check(step(-position, other), moved_position, moved_rotation, 0.0)
    # A second noncommuting delta composes from the held full pose, without a calibration jump.
    delta2 = Rotation.from_rotvec([0.2, 0.0, 0.0])
    check(step(-position + displacement, delta2 * other), moved_position + displacement, delta2 * moved_rotation, 0.0)
    check(step(position, origin, squeeze=0.0, reset=True), home_position, home_rotation, 0.0)
    check(step(-position, other), home_position, home_rotation, 0.0)
