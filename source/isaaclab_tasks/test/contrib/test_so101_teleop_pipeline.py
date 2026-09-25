# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for the SO-101 XR teleop pipeline wiring.

These tests build the IsaacTeleop pipeline standalone (no ``gym.make``, USD, GPU, or XR
device) and verify the flattened action width and element order.

The retargeter math itself (trigger -> closedness, clutch rebasing, orientation calibration) is
unit tested in Isaac Teleop (``test_so101_retargeters.py``); this file guards only the Lab-side
wiring:

- The pose retargeter emits a fixed 7-element output; the flattened action passes the full pose
  through and concatenates the gripper channel into the 8D action
  ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]``.
- Each entry in ``output_order`` must resolve to a declared element name; an undeclared
  name resolves silently to a constant 0.0 instead of raising.
"""

import pytest

pytest.importorskip("isaacteleop")

from isaacteleop.retargeters.SO101.gripper_retargeter import GRIPPER_ELEMENT_LABEL

from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import _build_so101_stack_pipeline


def test_so101_pipeline_action_is_8d():
    """The SO-101 pipeline flattens to an 8D action ``[pos_xyz, quat_xyzw, gripper]``."""
    combiner = _build_so101_stack_pipeline()

    action_type = combiner.output_types()["action"].types[0]
    assert action_type.shape == (8,), f"expected an 8D action, got shape {action_type.shape}"


def test_so101_pipeline_output_order():
    """The action elements resolve to the declared labels in order (not silent 0.0 fillers)."""
    combiner = _build_so101_stack_pipeline()

    # Walk the OutputCombiner -> subgraph -> TensorReorderer to read the resolved output order.
    # This depends on isaacteleop internals (no public accessor); fail rather than skip if they change,
    # since this is the only check of the element order.
    reorderer = combiner.output_mapping["action"].module._target_module
    output_order = reorderer._output_order

    expected_order = [
        "pos_x",
        "pos_y",
        "pos_z",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
        GRIPPER_ELEMENT_LABEL,
    ]
    assert output_order == expected_order, (
        f"unexpected output order {output_order}; the pose/gripper elements must resolve to the"
        f" declared pos/quat/{GRIPPER_ELEMENT_LABEL!r} labels so they do not become silent 0.0"
        " fillers"
    )
