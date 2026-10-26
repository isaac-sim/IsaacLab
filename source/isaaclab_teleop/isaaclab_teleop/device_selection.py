# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve and create script teleop devices, translating legacy configs to Isaac Capture."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from isaaclab.devices import DeviceCfg
from isaaclab.devices.gamepad import Se2GamepadCfg, Se3GamepadCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg, Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se2SpaceMouseCfg, Se3SpaceMouseCfg

from .device_pipelines import (
    se2_gamepad_teleop_cfg,
    se2_keyboard_teleop_cfg,
    se2_spacemouse_teleop_cfg,
    se3_gamepad_teleop_cfg,
    se3_keyboard_teleop_cfg,
    se3_spacemouse_teleop_cfg,
)
from .isaac_teleop_cfg import IsaacTeleopCfg

# Each device config type, the builder of its pipeline, and the config fields the builder takes.
_PIPELINES = {
    Se3KeyboardCfg: (se3_keyboard_teleop_cfg, ("pos_sensitivity", "rot_sensitivity", "gripper_term")),
    Se2KeyboardCfg: (se2_keyboard_teleop_cfg, ("v_x_sensitivity", "v_y_sensitivity", "omega_z_sensitivity")),
    Se3GamepadCfg: (se3_gamepad_teleop_cfg, ("pos_sensitivity", "rot_sensitivity", "dead_zone", "gripper_term")),
    Se2GamepadCfg: (
        se2_gamepad_teleop_cfg,
        ("v_x_sensitivity", "v_y_sensitivity", "omega_z_sensitivity", "dead_zone"),
    ),
    Se3SpaceMouseCfg: (se3_spacemouse_teleop_cfg, ("pos_sensitivity", "rot_sensitivity", "gripper_term")),
    Se2SpaceMouseCfg: (se2_spacemouse_teleop_cfg, ("v_x_sensitivity", "v_y_sensitivity", "omega_z_sensitivity")),
}


def _require_isaaccapture() -> None:
    if importlib.util.find_spec("isaacteleop") is None:
        raise ImportError(
            "Keyboard, gamepad and SpaceMouse teleoperation runs through Isaac Capture, which is not installed."
            " Install Isaac Lab's teleop extra (e.g. `uv sync --extra teleop`); it is available on Linux."
        )


def select_teleop_cfg(
    env_cfg: Any,
    device_name: str | None,
    default_device_cfgs: Mapping[str, IsaacTeleopCfg | DeviceCfg],
    sim_device: str,
) -> IsaacTeleopCfg | DeviceCfg:
    """Resolve a device once before scene startup.

    An unnamed request prefers the environment's Isaac Capture pipeline, then its keyboard or the script's default.
    A named request prefers the environment's device, then the script's built-ins. Keyboard, gamepad and
    SpaceMouse resolve to their Isaac Capture pipeline; devices without one (e.g. OpenXR hand tracking) keep
    their native config. Unknown names raise before launching the simulation.

    Raises:
        ImportError: If the request runs through Isaac Capture and it is not installed.
        ValueError: If the requested device is unknown.
    """
    if device_name is None:
        declared = getattr(env_cfg, "isaac_teleop", None)
        if declared is not None:
            _require_isaaccapture()
            return declared
        device_name = "keyboard"
    devices = getattr(env_cfg, "teleop_devices", None)
    device_cfg = devices.devices.get(device_name) if devices is not None else None
    if device_cfg is None:
        device_cfg = default_device_cfgs.get(device_name.lower())
    if device_cfg is None:
        raise ValueError(f"Unknown teleop device: {device_name!r}")
    if isinstance(device_cfg, IsaacTeleopCfg):
        _require_isaaccapture()
        return device_cfg
    entry = next((entry for cfg_type, entry in _PIPELINES.items() if isinstance(device_cfg, cfg_type)), None)
    if entry is None:
        return device_cfg
    _require_isaaccapture()
    builder, fields = entry
    warnings.warn(
        f"{type(device_cfg).__name__} is deprecated in Isaac Lab 3.2. "
        f"Use isaaclab_teleop.{builder.__name__}(...) instead.",
        FutureWarning,
        stacklevel=2,
    )
    cfg = builder(**{field: getattr(device_cfg, field) for field in fields}, sim_device=sim_device)
    cfg.teleoperation_active_default = device_cfg.teleoperation_active_default
    return cfg


def create_teleop_device(
    cfg: IsaacTeleopCfg | DeviceCfg, callbacks: Mapping[str, Callable] | None = None, **capture_options
):
    """Create the resolved device; Isaac Capture errors propagate instead of switching implementations.

    The existing native factory continues to construct native retargeters and register callbacks.
    Isaac Capture options, including the explicit visualizer sequence, apply only to Isaac Capture devices.
    """
    if isinstance(cfg, IsaacTeleopCfg):
        from .isaac_teleop_device import create_isaac_teleop_device

        return create_isaac_teleop_device(cfg, callbacks=callbacks, **capture_options)
    from .deprecated.teleop_device_factory import create_teleop_device as create_native_device

    name = type(cfg).__name__
    return create_native_device(name, {name: cfg}, callbacks)
