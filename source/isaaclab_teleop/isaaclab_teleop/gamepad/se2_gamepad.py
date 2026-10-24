# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad-driven IsaacTeleopCfg builder for SE(2) control."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..isaac_teleop_cfg import IsaacTeleopCfg


def se2_gamepad_teleop_cfg(
    v_x_sensitivity: float = 1.0,
    v_y_sensitivity: float = 1.0,
    omega_z_sensitivity: float = 1.0,
    dead_zone: float = 0.15,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    r"""Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for gamepad-driven SE(2) velocity control.

    Raw stick state is retargeted via :class:`~isaacteleop.retargeters.GamepadToSe2Retargeter`.

    The gamepad is read by the Isaac Capture ``gamepad`` plugin from the first connected Linux
    joystick-API device (``/dev/input/jsN``); the config starts the plugin with the session, so no
    headset or separate process is needed, and udev grants the logged-in user access without the
    ``input`` group.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      left stick up             left stick down
        Move along y-axis      left stick right          left stick left
        Rotate along z-axis    right stick right         right stick left
        ====================== ========================= ========================

    No physical button drives session start/stop/reset -- those come from the teleop session's
    control channel or an auxiliary control device, matching the legacy ``Se2Gamepad``.

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        dead_zone: Stick deflection magnitude below which input is treated as zero.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from .._plugins import bundled_plugin_configs
    from ..isaac_teleop_cfg import IsaacTeleopCfg

    def build_pipeline():
        from isaacteleop.retargeters import GamepadToSe2Retargeter, GamepadToSe2RetargeterConfig, TensorReorderer
        from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner

        gamepad_source = GamepadSource("gamepad")

        se2 = GamepadToSe2Retargeter(
            GamepadToSe2RetargeterConfig(
                v_x_sensitivity=v_x_sensitivity,
                v_y_sensitivity=v_y_sensitivity,
                omega_z_sensitivity=omega_z_sensitivity,
                dead_zone=dead_zone,
            ),
            name="se2",
        )
        connected_se2 = se2.connect({"gamepad": gamepad_source.output("gamepad")})

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
        app_name="IsaacLabGamepadSe2",
        plugins=bundled_plugin_configs("gamepad"),
    )
