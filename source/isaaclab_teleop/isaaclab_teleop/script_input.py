# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard hotkeys and teleop devices for scripts, through the visualizer windows and Isaac Capture."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from .device_selection import _isaacteleop_available, pipeline_from_device_cfg

if TYPE_CHECKING:
    from isaaclab.devices import DeviceCfg

logger = logging.getLogger(__name__)


def add_key_callbacks(callbacks: Mapping[str, Callable[[], None]]) -> Callable[[], None]:
    """Call ``callbacks[key]`` when ``key`` is pressed in a visualizer window, e.g. for a script's hotkeys.

    Keys are letters or digits (``"N"``, ``"1"``) or W3C ``KeyboardEvent.code`` strings (``"Escape"``).
    Key repeat is not a press. Reads every visualizer window open when called; the callbacks end when
    the visualizers close, or when the returned function is called.

    Args:
        callbacks: The function to call for each key. Takes no arguments.

    Returns:
        A function that removes the callbacks.
    """
    from isaaclab.sim import SimulationContext

    by_code = {_w3c_code(key): func for key, func in callbacks.items()}
    sim = SimulationContext.instance()
    subscriptions = []
    for visualizer in sim.visualizers if sim is not None else ():
        source = visualizer.key_event_source
        if source is None:
            continue
        held: set[str] = set()

        def on_key(code: str, pressed: bool, held: set[str] = held) -> None:
            if not pressed:
                held.discard(code)
            elif code not in held:
                held.add(code)
                if code in by_code:
                    by_code[code]()

        subscriptions.append(source.add_key_listener(on_key, held.clear))
    if not subscriptions:
        logger.info("No visualizer window reports keys, so the key callbacks are inactive")

    def remove() -> None:
        for subscription in subscriptions:
            subscription.close()

    return remove


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


def _w3c_code(key: str) -> str:
    if len(key) == 1 and key.isalpha():
        return f"Key{key.upper()}"
    if len(key) == 1 and key.isdigit():
        return f"Digit{key}"
    return key
