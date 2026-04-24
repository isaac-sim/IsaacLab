# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first dual-access array wrapper with explicit ``.torch`` and ``.warp`` accessors.

Inspired by ProxyArray from mujocolab/mjlab (BSD-3-Clause).
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import torch
import warp as wp


class ProxyArray:
    """Warp-first array wrapper providing cached zero-copy ``.torch`` and ``.warp`` accessors.

    This class wraps a :class:`warp.array` and provides:

    * A ``.warp`` property that returns the original warp array (for kernel interop).
    * A ``.torch`` property that returns a cached, zero-copy :class:`torch.Tensor` view
      (via :func:`warp.to_torch`).
    * Convenience properties (``shape``, ``dtype``, ``device``) delegated to the warp array.
    * A deprecation bridge (``__torch_function__`` and arithmetic/comparison operators) that
      allows existing code using ``ProxyArray`` as if it were a ``torch.Tensor`` to keep working
      while emitting a one-time :class:`DeprecationWarning`.

    Example:

    .. code-block:: python

        import warp as wp
        from isaaclab.utils.warp.proxy_array import ProxyArray

        arr = wp.zeros(100, dtype=wp.vec3f, device="cuda:0")
        ta = ProxyArray(arr)

        # Explicit access (preferred)
        ta.warp  # -> wp.array, shape (100,), dtype vec3f
        ta.torch  # -> torch.Tensor, shape (100, 3)

        # Deprecation bridge (warns once, then silent)
        result = ta + 1.0  # works, emits DeprecationWarning
    """

    _deprecation_warned: ClassVar[bool] = False
    """Class-level flag ensuring the deprecation warning is emitted at most once."""

    def __init__(self, wp_array: wp.array) -> None:
        """Initialize the ProxyArray wrapper.

        Args:
            wp_array: The warp array to wrap.

        Raises:
            TypeError: If ``wp_array`` is not a :class:`warp.array`.
        """
        if not isinstance(wp_array, wp.array):
            raise TypeError(
                f"ProxyArray expects a warp.array, got {type(wp_array).__name__}."
                " If you have a ProxyArray, use it directly instead of wrapping it again."
            )
        self._warp = wp_array
        self._torch_cache: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def rebind(self, wp_array: wp.array) -> None:
        """Rebind this wrapper to a new warp array, invalidating the torch cache.

        This is needed when the underlying simulation memory is re-created (e.g. after
        a full simulation reset) and the old warp array pointer becomes stale.

        Args:
            wp_array: The new warp array to wrap.
        """
        self._warp = wp_array
        self._torch_cache = None

    @property
    def warp(self) -> wp.array:
        """The underlying warp array."""
        return self._warp

    @property
    def torch(self) -> torch.Tensor:
        """A cached, zero-copy :class:`torch.Tensor` view of the warp array.

        The tensor is created on first access via :func:`warp.to_torch` and cached
        for subsequent calls. Since this is a zero-copy view, modifications to the
        tensor are visible through the warp array and vice versa.
        """
        if self._torch_cache is None:
            self._torch_cache = wp.to_torch(self._warp)
        return self._torch_cache

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying warp array."""
        return self._warp.shape

    @property
    def dtype(self):
        """Warp dtype of the underlying array."""
        return self._warp.dtype

    @property
    def device(self) -> str:
        """Device string of the underlying warp array."""
        return self._warp.device

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        return self._warp.shape[0]

    def __repr__(self) -> str:
        """Return a string representation of the ProxyArray."""
        return f"ProxyArray(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    # ------------------------------------------------------------------
    # Warp kernel interop
    # ------------------------------------------------------------------

    @property
    def __cuda_array_interface__(self):
        """Delegate the CUDA array interface to the underlying warp array.

        This allows a ``ProxyArray`` to be passed directly as an argument to
        :func:`warp.launch` without explicitly accessing ``.warp``.

        Raises:
            AttributeError: If the underlying warp array is not on a CUDA device.
        """
        return self._warp.__cuda_array_interface__

    @property
    def __array_interface__(self):
        """Delegate the NumPy array interface to the underlying warp array.

        This allows a ``ProxyArray`` to be passed directly as an argument to
        :func:`warp.launch` on CPU without explicitly accessing ``.warp``.

        Raises:
            AttributeError: If the underlying warp array is not on a CPU device.
        """
        return self._warp.__array_interface__

    # ------------------------------------------------------------------
    # Indexing (deprecation bridge — delegates to .torch)
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """Index into the torch view of this array.

        Supports all torch indexing: ``int``, ``slice``, ``tuple``,
        boolean masks, and fancy indexing (multi-dimensional).
        """
        self._warn_implicit()
        return self.torch[key]

    def __setitem__(self, key, value):
        """Write through the torch view into the shared warp memory.

        Supports all torch indexing: ``int``, ``slice``, ``tuple``,
        boolean masks, and fancy indexing (multi-dimensional).
        """
        self._warn_implicit()
        self.torch[key] = value

    # ------------------------------------------------------------------
    # Deprecation bridge
    # ------------------------------------------------------------------

    @classmethod
    def _warn_implicit(cls) -> None:
        """Emit a one-time deprecation warning for implicit torch usage."""
        if not cls._deprecation_warned:
            cls._deprecation_warned = True
            warnings.warn(
                "Implicit use of ProxyArray as a torch.Tensor is deprecated. "
                "Use the explicit .torch property instead (e.g., array.torch).",
                DeprecationWarning,
                stacklevel=3,
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Enable torch operations on ProxyArray by unwrapping to ``.torch``.

        This method is called by PyTorch when a torch function receives a
        ``ProxyArray`` as an argument. It unwraps all ``ProxyArray`` instances
        to their ``.torch`` tensors and delegates to the original function.
        """
        if kwargs is None:
            kwargs = {}
        cls._warn_implicit()

        def unwrap(x):
            if isinstance(x, ProxyArray):
                return x.torch
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(i) for i in x)
            return x

        args = unwrap(args)
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def _binop(self, other, op: str) -> torch.Tensor:
        """Helper for binary and reflected binary operations."""
        self._warn_implicit()
        other_val = other.torch if isinstance(other, ProxyArray) else other
        return getattr(self.torch, op)(other_val)

    def __add__(self, other) -> torch.Tensor:
        return self._binop(other, "__add__")

    def __radd__(self, other) -> torch.Tensor:
        return self._binop(other, "__radd__")

    def __sub__(self, other) -> torch.Tensor:
        return self._binop(other, "__sub__")

    def __rsub__(self, other) -> torch.Tensor:
        return self._binop(other, "__rsub__")

    def __mul__(self, other) -> torch.Tensor:
        return self._binop(other, "__mul__")

    def __rmul__(self, other) -> torch.Tensor:
        return self._binop(other, "__rmul__")

    def __truediv__(self, other) -> torch.Tensor:
        return self._binop(other, "__truediv__")

    def __rtruediv__(self, other) -> torch.Tensor:
        return self._binop(other, "__rtruediv__")

    def __pow__(self, other) -> torch.Tensor:
        return self._binop(other, "__pow__")

    def __rpow__(self, other) -> torch.Tensor:
        return self._binop(other, "__rpow__")

    def __neg__(self) -> torch.Tensor:
        self._warn_implicit()
        return -self.torch

    def __pos__(self) -> torch.Tensor:
        self._warn_implicit()
        return +self.torch

    def __abs__(self) -> torch.Tensor:
        self._warn_implicit()
        return abs(self.torch)

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def __eq__(self, other) -> torch.Tensor:
        return self._binop(other, "__eq__")

    def __ne__(self, other) -> torch.Tensor:
        return self._binop(other, "__ne__")

    def __lt__(self, other) -> torch.Tensor:
        return self._binop(other, "__lt__")

    def __le__(self, other) -> torch.Tensor:
        return self._binop(other, "__le__")

    def __gt__(self, other) -> torch.Tensor:
        return self._binop(other, "__gt__")

    def __ge__(self, other) -> torch.Tensor:
        return self._binop(other, "__ge__")
