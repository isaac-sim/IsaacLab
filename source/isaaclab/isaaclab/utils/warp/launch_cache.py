# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

import warp as wp


class _WarpLaunchCache:
    """Cache stable eager-CUDA commands returned by a Warp launch operation.

    The operation must accept ``device`` and ``record_cmd`` like ``wp.launch``.
    Its arguments and array storage must remain stable until the key is cleared.
    CPU execution and graph capture invoke the operation directly.
    """

    def __init__(self, device: str):
        self._device = wp.get_device(device)
        self._commands: dict[object, wp.Launch] = {}

    def launch(self, key: object, operation: Callable, *args, **kwargs) -> None:
        if not self._device.is_cuda or self._device.is_capturing:
            operation(*args, **kwargs, device=self._device)
            return
        command = self._commands.get(key)
        if command is None:
            command = operation(*args, **kwargs, device=self._device, record_cmd=True)
            if command is None:
                return
            self._commands[key] = command
        command.launch()

    def clear(self, key: object | None = None) -> None:
        """Remove one recorded launch or every recorded launch."""
        if key is None:
            self._commands.clear()
        else:
            self._commands.pop(key, None)
