# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cached Warp kernel launches for low-overhead record-and-replay execution."""

from __future__ import annotations

import operator
import os
import struct
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import warp as wp

_WARP_LAUNCH_MODE_ENV = "ISAACLAB_WARP_LAUNCH_MODE"
_WARP_LAUNCH_DEBUG_ENV = "ISAACLAB_WARP_LAUNCH_DEBUG"


class WarpLaunchCache:
    """Launch Warp kernels eagerly or replay a command recorded on first use.

    This utility is the per-kernel counterpart to a CUDA graph cache. Each
    ``(kernel, site)`` pair is recorded independently, so pointer-stable kernels
    can replay even when neighboring work cannot be captured or recorded. The
    launch interface contains only the arguments used by persistent Isaac Lab
    kernels.

    Recorded arguments are static. Their storage, layout, and scalar values must
    remain unchanged for the lifetime of the cached command. Use a distinct
    ``site`` when one owner invokes the same kernel with multiple persistent
    argument sets. Leave launches with changing argument storage on
    :func:`warp.launch`.

    The launch dimension is checked before replay. Debug mode additionally checks
    the static-argument contract on every replay. It is meant for tests and
    validation; normal replay only performs the command lookup, an optional
    dimension update, and :meth:`warp.Launch.launch`.

    Recorded launches may be replayed inside CUDA graph capture. CUDA graph
    nodes retain their own launch data, so the Python command need not outlive
    the graph. The argument storage is still caller-owned and must remain alive
    and pointer-stable until every graph that references it is invalidated and
    drained. Invalidate those graphs before resetting this cache if it is the
    only remaining owner of their argument storage.

    Args:
        mode: Execution mode. ``"auto"`` replays commands on CUDA and launches
            eagerly on CPU, ``"replay"`` requires command replay, and
            ``"eager"`` calls :func:`warp.launch` directly. If omitted, reads
            ``ISAACLAB_WARP_LAUNCH_MODE`` and defaults to ``"auto"``.
        debug: Whether to validate recorded arguments on every replay. If
            omitted, reads ``ISAACLAB_WARP_LAUNCH_DEBUG`` and defaults to
            ``False``.
        device: Fixed device for the cache owner.
    """

    @dataclass(frozen=True)
    class _ArrayDescriptor:
        """Pointer and layout properties of one array argument."""

        ptr: int
        grad_ptr: int
        device: str
        dtype: str
        shape: tuple[int, ...]
        strides: tuple[int, ...]

    @dataclass
    class _Entry:
        """One recorded command and its static launch contract."""

        command: wp.Launch
        site: Hashable | None
        dim: tuple[int, ...]
        argument_counts: tuple[int, int]
        argument_tokens: tuple[tuple[Any, ...], ...] | None
        argument_owners: tuple[Any, ...]

    def __init__(
        self,
        mode: Literal["auto", "eager", "replay"] | None = None,
        debug: bool | None = None,
        *,
        device: str | wp.Device,
    ):
        if mode is None:
            mode = os.environ.get(_WARP_LAUNCH_MODE_ENV, "auto")
        mode = mode.lower()
        if mode not in {"auto", "eager", "replay"}:
            raise ValueError(
                f"Invalid Warp launch mode {mode!r}. Set {_WARP_LAUNCH_MODE_ENV} to 'auto', 'eager', or 'replay'."
            )

        if debug is None:
            debug = self._read_debug_environment()

        self._mode = mode
        self._debug = debug
        self._device = wp.get_device(device)
        self._entries: dict[Hashable, WarpLaunchCache._Entry] = {}

    def launch(
        self,
        kernel: wp.Kernel,
        dim: int | Sequence[int],
        inputs: Sequence[Any] = (),
        outputs: Sequence[Any] = (),
        *,
        stream: wp.Stream | None = None,
        site: Hashable | None = None,
    ) -> None:
        """Launch a Warp kernel eagerly or through a cached command.

        Args:
            kernel: Warp kernel to launch.
            dim: Number or shape of threads to launch. A zero-sized launch is a
                no-op and is not cached.
            inputs: Forward input arguments.
            outputs: Forward output arguments.
            stream: Optional stream on the owner's device.
            site: Optional stable identity for one persistent argument set.

        Raises:
            TypeError: If ``site`` is not hashable.
            ValueError: If the launch dimension is invalid.
            RuntimeError: In debug mode, if a recorded static argument changes.
        """
        if self._mode == "eager" or (self._mode == "auto" and not self._device.is_cuda):
            wp.launch(
                kernel,
                dim=dim,
                inputs=inputs,
                outputs=outputs,
                device=self._device,
                stream=stream,
            )
            return

        logical_key: Hashable = kernel if site is None else (kernel, site)
        try:
            entry = self._entries.get(logical_key)
        except TypeError as exc:
            raise TypeError(f"Warp launch site must be hashable, got {type(site).__name__}.") from exc

        if entry is not None:
            if isinstance(dim, int):
                dim_matches = len(entry.dim) == 1 and dim == entry.dim[0]
            else:
                dim_matches = isinstance(dim, tuple) and dim == entry.dim
            if not dim_matches:
                normalized_dim, total_dim = self._normalize_dim(dim)
                if total_dim == 0:
                    return
                if normalized_dim != entry.dim:
                    entry.command.set_dim(normalized_dim)
                    entry.dim = normalized_dim
            if self._debug:
                self._validate_entry(entry, inputs, outputs)
            entry.command.launch(stream=stream)
            return

        normalized_dim, total_dim = self._normalize_dim(dim)
        if total_dim == 0:
            return
        command = wp.launch(
            kernel,
            dim=normalized_dim,
            inputs=inputs,
            outputs=outputs,
            device=self._device,
            stream=stream,
            record_cmd=True,
        )
        if command is None:
            raise RuntimeError(f"Warp did not return a Launch while recording {logical_key!r} on {self._device}.")

        arguments = (*inputs, *outputs)
        entry = self._Entry(
            command=command,
            site=site,
            dim=normalized_dim,
            argument_counts=(len(inputs), len(outputs)),
            argument_tokens=self._argument_tokens(arguments) if self._debug else None,
            argument_owners=arguments,
        )
        self._entries[logical_key] = entry
        command.launch(stream=stream)

    def invalidate(self, site: Hashable | None = None) -> None:
        """Discard recorded commands.

        Args:
            site: Optional site to invalidate across kernels. If omitted, every
                command is discarded.

        Invalidate captured graphs that reference these commands before calling
        this method if the cache is their argument storage's only remaining
        owner. Outstanding work on the owner device is synchronized before the
        cached commands' retained argument owners are released. An empty cache
        does not synchronize.
        """
        if site is None:
            self.reset()
            return
        stale_keys = [key for key, entry in self._entries.items() if self._sites_match(entry.site, site)]
        if stale_keys:
            wp.synchronize_device(self._device)
        for key in stale_keys:
            del self._entries[key]

    def reset(self) -> None:
        """Drain dependent work and discard every recorded command.

        Invalidate captured graphs that reference these commands before calling
        this method if the cache is their argument storage's only remaining
        owner. The owner device is synchronized only when the cache contains
        recorded commands. An empty cache returns immediately.
        """
        if self._entries:
            wp.synchronize_device(self._device)
        self._entries.clear()

    @staticmethod
    def _sites_match(recorded: Hashable | None, requested: Hashable) -> bool:
        """Compare site keys without forcing array-like equality to a boolean."""
        if recorded is requested:
            return True
        try:
            if hash(recorded) != hash(requested):
                return False
            return bool(recorded == requested)
        except (RuntimeError, TypeError, ValueError):
            return False

    def _validate_entry(
        self,
        entry: WarpLaunchCache._Entry,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
    ) -> None:
        """Validate the static command contract in debug mode."""
        counts = (len(inputs), len(outputs))
        if counts != entry.argument_counts:
            raise RuntimeError(
                f"Warp launch argument counts changed: recorded {entry.argument_counts}, received {counts}. "
                "Use a distinct site or invalidate the cache."
            )
        arguments = (*inputs, *outputs)
        tokens = self._argument_tokens(arguments)
        if tokens != entry.argument_tokens:
            mismatch = next(
                index for index, pair in enumerate(zip(entry.argument_tokens or (), tokens)) if pair[0] != pair[1]
            )
            raise RuntimeError(
                f"Warp launch static argument {mismatch} changed. Recorded commands require persistent argument "
                "storage and values; use a distinct site, invalidate the cache, or leave this launch eager."
            )

    @staticmethod
    def _normalize_dim(dim: int | Sequence[int]) -> tuple[tuple[int, ...], int]:
        """Normalize Warp launch dimensions and compute their product."""
        if isinstance(dim, int):
            normalized = (dim,)
        else:
            try:
                normalized = tuple(operator.index(extent) for extent in dim)
            except TypeError:
                normalized = (operator.index(dim),)

        if len(normalized) > 4:
            raise ValueError(f"Warp launch dimensions must have at most 4 axes, got {len(normalized)}.")

        total = 1
        for extent in normalized:
            if extent < 0:
                raise ValueError(f"Warp launch dimensions must be non-negative, got {normalized}.")
            total *= extent
        return normalized, total

    @classmethod
    def _argument_tokens(cls, arguments: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
        """Snapshot static arguments for debug validation."""
        return tuple(cls._argument_token(value) for value in arguments)

    @classmethod
    def _argument_token(cls, value: Any) -> tuple[Any, ...]:
        """Return a stable pointer, layout, or value token for one argument."""
        descriptor = cls._array_descriptor_or_none(value)
        if descriptor is not None:
            return ("array", descriptor)
        if type(value) is float:
            return ("float", struct.pack("=d", value))
        if type(value) is complex:
            return ("complex", struct.pack("=dd", value.real, value.imag))
        if value is None or isinstance(value, (bool, int, str, bytes)):
            return ("value", type(value), value)
        if isinstance(value, tuple):
            return ("tuple", type(value), tuple(cls._argument_token(item) for item in value))
        if hasattr(type(value), "_wp_scalar_type_"):
            try:
                return ("warp_value", type(value), tuple(value))
            except TypeError:
                pass
        if hasattr(value, "value"):
            return ("value_attribute", type(value), value.value)
        return ("object", type(value), id(value))

    @classmethod
    def _array_descriptor_or_none(cls, value: Any) -> WarpLaunchCache._ArrayDescriptor | None:
        """Return a descriptor for a Warp array, ProxyArray, or Torch tensor."""
        array = value if isinstance(value, wp.array) else getattr(value, "warp", None)
        if isinstance(array, wp.array):
            grad = array.grad
            return cls._ArrayDescriptor(
                ptr=0 if array.ptr is None else int(array.ptr),
                grad_ptr=0 if grad is None or grad.ptr is None else int(grad.ptr),
                device=str(array.device),
                dtype=str(array.dtype),
                shape=tuple(int(extent) for extent in array.shape),
                strides=tuple(int(stride) for stride in array.strides),
            )

        data_ptr = getattr(value, "data_ptr", None)
        stride = getattr(value, "stride", None)
        element_size = getattr(value, "element_size", None)
        if callable(data_ptr) and callable(stride) and callable(element_size) and hasattr(value, "shape"):
            return cls._ArrayDescriptor(
                ptr=int(data_ptr()),
                grad_ptr=0,
                device=str(getattr(value, "device", "cpu")),
                dtype=str(getattr(value, "dtype", type(value))),
                shape=tuple(int(extent) for extent in value.shape),
                strides=tuple(int(value_stride) * int(element_size()) for value_stride in stride()),
            )
        return None

    @staticmethod
    def _read_debug_environment() -> bool:
        """Read the boolean debug switch from the environment."""
        value = os.environ.get(_WARP_LAUNCH_DEBUG_ENV, "0").lower()
        if value in {"0", "false", "no", "off"}:
            return False
        if value in {"1", "true", "yes", "on"}:
            return True
        raise ValueError(f"Invalid {_WARP_LAUNCH_DEBUG_ENV} value {value!r}; use 0/1, false/true, no/yes, or off/on.")
