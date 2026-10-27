# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard, gamepad and SpaceMouse pipelines for Isaac Capture with shared action composition."""

from __future__ import annotations

from .isaac_teleop_cfg import IsaacTeleopCfg


def _action_pipeline(motion, gripper=None, **outputs):
    """Expose a flat action, joining pose and gripper only when both are present."""
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    if gripper is not None:
        from isaacteleop.retargeters import TensorReorderer

        pose = ["dx", "dy", "dz", "drx", "dry", "drz"]
        motion = (
            TensorReorderer(
                input_config={"motion": pose, "gripper": ["gripper"]},
                output_order=pose + ["gripper"],
                name="action_reorderer",
                input_types={"motion": "array", "gripper": "scalar"},
            )
            .connect({"motion": motion, "gripper": gripper})
            .output("output")
        )
    return OutputCombiner({"action": motion, **outputs})


def se2_keyboard_teleop_cfg(
    v_x_sensitivity: float = 0.8,
    v_y_sensitivity: float = 0.4,
    omega_z_sensitivity: float = 1.0,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    """Build a keyboard pipeline for SE(2) velocity control.

    Use arrows/numpad 8/2 and 4/6 for translation, and Z/X or numpad 7/9 for rotation.

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """

    def build_pipeline():
        from isaacteleop.retargeters import KeyboardToSe2Retargeter, KeyboardToSe2RetargeterConfig
        from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource

        keyboard_source = KeyboardSource("keyboard")
        motion = KeyboardToSe2Retargeter(
            KeyboardToSe2RetargeterConfig(
                v_x_sensitivity=v_x_sensitivity,
                v_y_sensitivity=v_y_sensitivity,
                omega_z_sensitivity=omega_z_sensitivity,
            ),
            name="se2",
        ).connect({"keyboard_held": keyboard_source.output("keyboard_held")})
        return _action_pipeline(motion.output("base_command"))

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabKeyboardSe2",
    )


def se3_keyboard_teleop_cfg(
    pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, gripper_term: bool = True, sim_device: str = "cpu"
) -> IsaacTeleopCfg:
    """Build a keyboard pipeline for SE(3) delta-pose control.

    See :ref:`isaac-teleop-keyboard` for bindings, focus and device permissions.

    Args:
        pos_sensitivity: Position delta scale per step [m].
        rot_sensitivity: Rotation delta scale per step [rad].
        gripper_term: Whether to include a gripper open/close command as a 7th action element.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """

    def build_pipeline():
        from isaacteleop.retargeters import (
            KeyboardGripperRetargeter,
            KeyboardToSe3RelRetargeter,
            KeyboardToSe3RelRetargeterConfig,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource

        keyboard_source = KeyboardSource("keyboard")
        motion = KeyboardToSe3RelRetargeter(
            KeyboardToSe3RelRetargeterConfig(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity),
            name="se3",
        ).connect({"keyboard_held": keyboard_source.output("keyboard_held")})
        gripper = None
        if gripper_term:
            gripper = (
                KeyboardGripperRetargeter(name="gripper")
                .connect({"keyboard_pressed": keyboard_source.output("keyboard_pressed")})
                .output("gripper_command")
            )
        return _action_pipeline(motion.output("ee_delta"), gripper)

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabKeyboardSe3",
    )


def se2_gamepad_teleop_cfg(
    v_x_sensitivity: float = 1.0,
    v_y_sensitivity: float = 1.0,
    omega_z_sensitivity: float = 1.0,
    dead_zone: float = 0.15,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    """Build a gamepad pipeline for SE(2) velocity control.

    See :ref:`isaac-teleop-keyboard` for bindings, focus and device permissions.

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        dead_zone: Stick deflection magnitude below which input is treated as zero.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR, PluginConfig

    def build_pipeline():
        from isaacteleop.retargeters import GamepadToSe2Retargeter, GamepadToSe2RetargeterConfig
        from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource

        gamepad_source = GamepadSource("gamepad")
        motion = GamepadToSe2Retargeter(
            GamepadToSe2RetargeterConfig(
                v_x_sensitivity=v_x_sensitivity,
                v_y_sensitivity=v_y_sensitivity,
                omega_z_sensitivity=omega_z_sensitivity,
                dead_zone=dead_zone,
            ),
            name="se2",
        ).connect({"gamepad": gamepad_source.output("gamepad")})
        return _action_pipeline(motion.output("base_command"))

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabGamepadSe2",
        plugins=[
            PluginConfig(
                plugin_name="gamepad", plugin_root_id="gamepad", search_paths=[BUNDLED_PLUGINS_DIR], required=True
            )
        ],
    )


def se3_gamepad_teleop_cfg(
    pos_sensitivity: float = 1.0,
    rot_sensitivity: float = 1.6,
    dead_zone: float = 0.15,
    gripper_term: bool = True,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    """Build a gamepad pipeline for SE(3) delta-pose control.

    See :ref:`isaac-teleop-keyboard` for bindings, focus and device permissions.

    Args:
        pos_sensitivity: Position delta scale per step [m].
        rot_sensitivity: Rotation delta scale per step [rad].
        dead_zone: Stick deflection magnitude below which input is treated as zero.
        gripper_term: Whether to include a gripper open/close command as a 7th action element.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR, PluginConfig

    def build_pipeline():
        from isaacteleop.retargeters import (
            GamepadGripperRetargeter,
            GamepadToSe3RelRetargeter,
            GamepadToSe3RelRetargeterConfig,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource

        gamepad_source = GamepadSource("gamepad")
        motion = GamepadToSe3RelRetargeter(
            GamepadToSe3RelRetargeterConfig(
                pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity, dead_zone=dead_zone
            ),
            name="se3",
        ).connect({"gamepad": gamepad_source.output("gamepad")})
        gripper = None
        if gripper_term:
            gripper = (
                GamepadGripperRetargeter(name="gripper")
                .connect({"gamepad": gamepad_source.output("gamepad")})
                .output("gripper_command")
            )
        return _action_pipeline(motion.output("ee_delta"), gripper)

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabGamepadSe3",
        plugins=[
            PluginConfig(
                plugin_name="gamepad", plugin_root_id="gamepad", search_paths=[BUNDLED_PLUGINS_DIR], required=True
            )
        ],
    )


def se2_spacemouse_teleop_cfg(
    v_x_sensitivity: float = 0.8,
    v_y_sensitivity: float = 0.4,
    omega_z_sensitivity: float = 1.0,
    sim_device: str = "cpu",
) -> IsaacTeleopCfg:
    """Build a spacemouse pipeline for SE(2) velocity control.

    See :ref:`isaac-teleop-keyboard` for bindings, focus and device permissions.

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR, PluginConfig

    def build_pipeline():
        from isaacteleop.retargeters import SpaceMouseToSe2Retargeter, SpaceMouseToSe2RetargeterConfig
        from isaacteleop.retargeting_engine.deviceio_source_nodes import SpaceMouseSource

        spacemouse_source = SpaceMouseSource("spacemouse")
        motion = SpaceMouseToSe2Retargeter(
            SpaceMouseToSe2RetargeterConfig(
                v_x_sensitivity=v_x_sensitivity,
                v_y_sensitivity=v_y_sensitivity,
                omega_z_sensitivity=omega_z_sensitivity,
            ),
            name="se2",
        ).connect(
            {
                "spacemouse_translation": spacemouse_source.output("spacemouse_translation"),
                "spacemouse_rotation": spacemouse_source.output("spacemouse_rotation"),
            }
        )
        return _action_pipeline(
            motion.output("base_command"), spacemouse_buttons=spacemouse_source.output("spacemouse_buttons")
        )

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabSpaceMouseSe2",
        plugins=[
            PluginConfig(
                plugin_name="spacemouse", plugin_root_id="spacemouse", search_paths=[BUNDLED_PLUGINS_DIR], required=True
            )
        ],
    )


def se3_spacemouse_teleop_cfg(
    pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, gripper_term: bool = True, sim_device: str = "cpu"
) -> IsaacTeleopCfg:
    """Build a spacemouse pipeline for SE(3) delta-pose control.

    See :ref:`isaac-teleop-keyboard` for bindings, focus and device permissions.

    Args:
        pos_sensitivity: Position delta scale per step [m].
        rot_sensitivity: Rotation delta scale per step [rad].
        gripper_term: Whether to include a gripper open/close command as a 7th action element.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR, PluginConfig

    def build_pipeline():
        from isaacteleop.retargeters import (
            SpaceMouseGripperRetargeter,
            SpaceMouseToSe3RelRetargeter,
            SpaceMouseToSe3RelRetargeterConfig,
        )
        from isaacteleop.retargeting_engine.deviceio_source_nodes import SpaceMouseSource

        spacemouse_source = SpaceMouseSource("spacemouse")
        motion = SpaceMouseToSe3RelRetargeter(
            SpaceMouseToSe3RelRetargeterConfig(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity),
            name="se3",
        ).connect(
            {
                "spacemouse_translation": spacemouse_source.output("spacemouse_translation"),
                "spacemouse_rotation": spacemouse_source.output("spacemouse_rotation"),
            }
        )
        gripper = None
        if gripper_term:
            gripper = (
                SpaceMouseGripperRetargeter(name="gripper")
                .connect({"spacemouse_buttons": spacemouse_source.output("spacemouse_buttons")})
                .output("gripper_command")
            )
        return _action_pipeline(
            motion.output("ee_delta"), gripper, spacemouse_buttons=spacemouse_source.output("spacemouse_buttons")
        )

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabSpaceMouseSe3",
        plugins=[
            PluginConfig(
                plugin_name="spacemouse", plugin_root_id="spacemouse", search_paths=[BUNDLED_PLUGINS_DIR], required=True
            )
        ],
    )
