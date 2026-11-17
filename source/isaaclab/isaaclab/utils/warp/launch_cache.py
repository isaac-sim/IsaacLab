# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Any

import warp as wp

_LAUNCH_ARGUMENTS = ("kernel", "dim", "inputs", "outputs")
"""The arguments of ``wp.launch`` a cached launch compares one by one; any other argument is a launch option."""

_UNKNOWN = object()
"""Key of an argument whose value cannot be compared, such as a Warp struct: it is set on every launch."""


class _WarpLaunchCache:
    """Recorded Warp launches of code that runs outside a CUDA graph, one per call site.

    Each call site names itself with ``key``, so the cache holds at most one :class:`warp.Launch` per call site. A
    launch compares its arguments with those of the call site's cached launch:

    * Through ``wp.launch``, a changed kernel argument is set on the cached launch in place, and so is a changed launch
      dimension. A kernel argument that cannot be compared, such as a Warp struct, is set on every launch. A changed
      kernel or launch option records the call site's launch again.
    * Through any other operation, which must accept ``device`` and ``record_cmd`` like ``wp.launch``, a changed array
      or scalar argument records the call site's launch again; other arguments must stay valid until the key is
      cleared.

    CPU execution and graph capture invoke the operation directly.
    """

    def __init__(self, device: str):
        self._device = wp.get_device(device)
        self._commands: dict[object, tuple[wp.Launch, tuple]] = {}

    def launch(self, key: object, operation: Callable, *args, **kwargs) -> None:
        """Launch through the call site's cached launch, recording it on the first call.

        Args:
            key: Name of the call site.
            operation: ``wp.launch``, or an operation that accepts ``device`` and ``record_cmd`` like it.
            *args: Positional arguments of the operation.
            **kwargs: Keyword arguments of the operation, without ``device``.
        """
        if not self._device.is_cuda or self._device.is_capturing:
            operation(*args, **kwargs, device=self._device)
            return
        cached = self._commands.get(key)
        if operation is wp.launch and len(args) <= len(_LAUNCH_ARGUMENTS):
            launch = dict(zip(_LAUNCH_ARGUMENTS, args), **kwargs)
            kernel, dim = launch.pop("kernel"), launch.pop("dim")
            params = [*launch.pop("inputs", ()), *launch.pop("outputs", ())]
            keys = [_key(value) for value in params]
            # the launch options left, such as block_dim, compared as a whole
            options = tuple(sorted((name, _key(value)) for name, value in launch.items()))
            if cached is not None:
                command, (cached_kernel, cached_dim, cached_keys, cached_options) = cached
                if cached_kernel is kernel and cached_options == options and len(cached_keys) == len(keys):
                    for index, (new, old) in enumerate(zip(keys, cached_keys)):
                        if new is _UNKNOWN or new != old:
                            command.set_param_at_index(index, params[index])
                    if dim != cached_dim:
                        command.set_dim(dim)
                    self._commands[key] = command, (kernel, dim, keys, options)
                    command.launch()
                    return
            signature = (kernel, dim, keys, options)
        else:
            signature = _key((args, kwargs), unknown=None)
            if cached is not None and cached[1] == signature:
                cached[0].launch()
                return
        command = operation(*args, **kwargs, device=self._device, record_cmd=True)
        if command is None:
            return
        self._commands[key] = command, signature
        command.launch()

    def clear(self, key: object | None = None) -> None:
        """Remove one recorded launch or every recorded launch."""
        if key is None:
            self._commands.clear()
        else:
            self._commands.pop(key, None)


def _key(value: Any, unknown: Any = _UNKNOWN) -> Any:
    """Key an argument by what a launch reads of it: an array by its pointer and layout, a scalar by its type and value.

    A :class:`~isaaclab.utils.warp.ProxyArray` keys as the Warp array it wraps. Lists and tuples key item by item. Any
    other value keys as ``unknown``.
    """
    if isinstance(value, wp.array):
        return value.ptr, value.shape, value.strides, value.dtype
    # True, 1 and 1.0 compare equal, so the type tells them apart
    if value is None or isinstance(value, (bool, int, float, str)):
        return type(value), value
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_key(item, unknown) for item in value)
    if isinstance(value, dict):
        return tuple((name, _key(item, unknown)) for name, item in value.items())
    array = getattr(value, "warp", None)
    if isinstance(array, wp.array):
        return _key(array)
    return unknown
