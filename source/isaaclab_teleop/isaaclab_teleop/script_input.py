# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop devices for scripts, through Isaac Capture when it is installed."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .device_selection import _isaacteleop_available, pipeline_from_device_cfg

if TYPE_CHECKING:
    from isaaclab.devices import DeviceCfg


@contextlib.contextmanager
def open_teleop_device(device_cfg: DeviceCfg) -> Iterator[Any]:
    """Open the keyboard, gamepad or SpaceMouse a device config describes, through Isaac Capture when installed.

    With the ``isaacteleop`` package the device runs as an Isaac Capture pipeline, started without a
    headset on the standalone CloudXR profile, and ``B`` / ``P`` / ``R`` start, pause and reset it (see
    :class:`~isaaclab_teleop.IsaacTeleopDevice`). Without it, the :mod:`isaaclab.devices` device is used.
    Either way ``advance()`` returns the device command (``None`` while an Isaac Capture session starts)
    and ``add_callback("R", func)`` registers a reset callback.

    Args:
        device_cfg: A keyboard, gamepad or SpaceMouse config, e.g. :class:`~isaaclab.devices.Se3KeyboardCfg`.

    Yields:
        The open device.
    """
    pipeline = pipeline_from_device_cfg(device_cfg, device_cfg.sim_device) if _isaacteleop_available() else None
    if pipeline is None:
        from isaaclab.utils import instantiate

        yield instantiate(device_cfg)
        return
    from .isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV
    from .isaac_teleop_device import create_isaac_teleop_device

    with create_isaac_teleop_device(
        pipeline, use_kit_xr_bridge=False, cloudxr_env_file=CLOUDXR_STANDALONE_ENV
    ) as device:
        device.request_start()
        yield device
