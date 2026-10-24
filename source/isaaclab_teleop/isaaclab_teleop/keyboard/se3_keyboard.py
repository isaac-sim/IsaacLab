# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard-driven IsaacTeleopCfg builder for SE(3) control."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..isaac_teleop_cfg import IsaacTeleopCfg


def se3_keyboard_teleop_cfg(
    pos_sensitivity: float = 0.4,
    rot_sensitivity: float = 0.8,
    gripper_term: bool = True,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    """Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for keyboard-driven SE(3) delta-pose control.

    Keys come from an in-process :class:`~isaacteleop.retargeting_engine.deviceio_source_nodes.KeyboardSource`,
    fed by whichever visualizer window has focus (see :ref:`isaac-teleop-keyboard`), and are
    retargeted via :class:`~isaacteleop.retargeters.KeyboardToSe3RelRetargeter` /
    :class:`~isaacteleop.retargeters.KeyboardGripperRetargeter`. The pipeline needs no plugin
    process or headset.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    ``B`` / ``P`` / ``R`` start, pause and reset teleoperation (see
    :class:`~isaaclab_teleop.IsaacTeleopDevice`).

    Args:
        pos_sensitivity: Position delta scale per step [m].
        rot_sensitivity: Rotation delta scale per step [rad].
        gripper_term: Whether to include a gripper open/close command as a 7th action element.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from ..isaac_teleop_cfg import IsaacTeleopCfg

    def build_pipeline():
        from isaacteleop.retargeters import (
            KeyboardGripperRetargeter,
            KeyboardToSe3RelRetargeter,
            KeyboardToSe3RelRetargeterConfig,
            TensorReorderer,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner

        keyboard_source = KeyboardSource("keyboard")

        se3 = KeyboardToSe3RelRetargeter(
            KeyboardToSe3RelRetargeterConfig(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity),
            name="se3",
        )
        connected_se3 = se3.connect({"keyboard_held": keyboard_source.output("keyboard_held")})

        ee_delta_elements = ["dx", "dy", "dz", "drx", "dry", "drz"]
        input_config = {"ee_delta": ee_delta_elements}
        input_types = {"ee_delta": "array"}
        reorder_inputs = {"ee_delta": connected_se3.output("ee_delta")}
        output_order = list(ee_delta_elements)

        if gripper_term:
            gripper = KeyboardGripperRetargeter(name="gripper")
            connected_gripper = gripper.connect({"keyboard_pressed": keyboard_source.output("keyboard_pressed")})
            input_config["gripper_command"] = ["gripper_value"]
            input_types["gripper_command"] = "scalar"
            reorder_inputs["gripper_command"] = connected_gripper.output("gripper_command")
            output_order.append("gripper_value")

        reorderer = TensorReorderer(
            input_config=input_config,
            output_order=output_order,
            name="action_reorderer",
            input_types=input_types,
        )
        connected_reorderer = reorderer.connect(reorder_inputs)

        return OutputCombiner({"action": connected_reorderer.output("output")})

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabKeyboardSe3",
    )
