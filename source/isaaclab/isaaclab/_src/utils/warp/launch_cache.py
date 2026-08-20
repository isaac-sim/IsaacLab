# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


class _WarpLaunchCache:
    """Cache stable eager-CUDA launches."""

    def __init__(self, device: str):
        self._device = wp.get_device(device)
        self._commands: dict[object, wp.Launch] = {}

    def launch(self, key: object, kernel: wp.Kernel, *, dim, inputs, outputs) -> None:
        if not self._device.is_cuda or self._device.is_capturing:
            wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=self._device)
            return
        command = self._commands.get(key)
        if command is None:
            command = wp.launch(
                kernel,
                dim=dim,
                inputs=inputs,
                outputs=outputs,
                device=self._device,
                record_cmd=True,
            )
            if command is None:
                return
            self._commands[key] = command
        command.launch()

    def clear(self) -> None:
        self._commands.clear()
