# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard-driven IsaacTeleopCfg builder for SE(2) control."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..isaac_teleop_cfg import IsaacTeleopCfg


def se2_keyboard_teleop_cfg(
    v_x_sensitivity: float = 0.8,
    v_y_sensitivity: float = 0.4,
    omega_z_sensitivity: float = 1.0,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    r"""Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for keyboard-driven SE(2) velocity control.

    Keys come from an in-process :class:`~isaacteleop.retargeting_engine.deviceio_source_nodes.KeyboardSource`,
    fed by whichever visualizer window has focus (see :ref:`isaac-teleop-keyboard`), and are
    retargeted via :class:`~isaacteleop.retargeters.KeyboardToSe2Retargeter`. The pipeline needs
    no plugin process or headset.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
        Move along y-axis      Numpad 4 / Arrow Left     Numpad 6 / Arrow Right
        Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
        ====================== ========================= ========================

    ``B`` / ``P`` / ``R`` start, pause and reset teleoperation (see
    :class:`~isaaclab_teleop.IsaacTeleopDevice`).

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from ..isaac_teleop_cfg import IsaacTeleopCfg

    def build_pipeline():
        from isaacteleop.retargeters import KeyboardToSe2Retargeter, KeyboardToSe2RetargeterConfig, TensorReorderer
        from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner

        keyboard_source = KeyboardSource("keyboard")

        se2 = KeyboardToSe2Retargeter(
            KeyboardToSe2RetargeterConfig(
                v_x_sensitivity=v_x_sensitivity,
                v_y_sensitivity=v_y_sensitivity,
                omega_z_sensitivity=omega_z_sensitivity,
            ),
            name="se2",
        )
        connected_se2 = se2.connect({"keyboard_held": keyboard_source.output("keyboard_held")})

        base_command_elements = ["v_x", "v_y", "omega_z"]
        reorderer = TensorReorderer(
            input_config={"base_command": base_command_elements},
            output_order=base_command_elements,
            name="action_reorderer",
            input_types={"base_command": "array"},
        )
        connected_reorderer = reorderer.connect({"base_command": connected_se2.output("base_command")})

        return OutputCombiner({"action": connected_reorderer.output("output")})

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabKeyboardSe2",
    )
