# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first dual-access array wrapper with explicit ``.torch`` and ``.warp`` accessors.

Inspired by ProxyArray from mujocolab/mjlab (BSD-3-Clause).
"""

from __future__ import annotations

import os
import warnings
from typing import ClassVar

import torch
import warp as wp

_QUATF_ACCESS_WARN_ENV = "WARN_ON_TORCH_QUATF_ACCESS"
"""Environment variable that, when set to ``"1"``, makes :attr:`ProxyArray.torch`
emit a :class:`UserWarning` on every read of a ``wp.quatf``-typed array. Used as a
runtime aid for tracking down call sites that may still assume Isaac Lab 2.x's
``(w, x, y, z)`` quaternion convention after the migration to Isaac Lab 3.x's
``(x, y, z, w)`` convention. See the Isaac Lab 3.0 migration guide for details."""


class ProxyArray:
    """Warp-first array wrapper providing cached zero-copy ``.torch`` and ``.warp`` accessors.

    This class wraps a :class:`warp.array` and provides:

    * A ``.warp`` property that returns the original warp array (for kernel interop).
    * A ``.torch`` property that returns a cached, zero-copy :class:`torch.Tensor` view
      (via :func:`warp.to_torch`).
    * Convenience properties (``shape``, ``dtype``, ``device``) delegated to the warp array.
    * A deprecation bridge for common torch functions, indexing, and arithmetic/comparison
      operators while emitting a one-time :class:`DeprecationWarning`. Tensor instance methods
      such as ``clone()`` are not forwarded; use explicit ``.torch`` access for those.

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

        The instance is immutable after construction: the wrapped ``wp.array`` cannot
        be reassigned. If the underlying simulation memory is re-allocated, construct
        a new :class:`ProxyArray` instead of mutating an existing one.

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
        # Bypass __setattr__ for the two internal fields — everything else raises.
        object.__setattr__(self, "_warp", wp_array)
        object.__setattr__(self, "_torch_cache", None)
        # Cached once at construction so the .torch read path stays a constant-time
        # check; only used when the WARN_ON_TORCH_QUATF_ACCESS env var is set.
        object.__setattr__(self, "_is_quatf", wp_array.dtype is wp.quatf)

    def __setattr__(self, name: str, value) -> None:
        """Forbid mutation of ProxyArray instances except for the internal torch cache.

        The torch view is populated lazily on first ``.torch`` access; that is the
        only allowed post-init state change. Every other write raises
        :class:`AttributeError` so callers don't accidentally re-point the wrapper.
        """
        if name == "_torch_cache":
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"ProxyArray is immutable; cannot set attribute {name!r}."
            " Construct a new ProxyArray instead of mutating an existing one."
        )

    @staticmethod
    def _quatf_access_warning_enabled() -> bool:
        """Return ``True`` when the ``WARN_ON_TORCH_QUATF_ACCESS`` env var is set to ``"1"``.

        Read on every :attr:`torch` access to keep the flag dynamic — a single
        ``os.environ`` lookup is cheap relative to the warp/torch interop work
        that follows.
        """
        return os.environ.get(_QUATF_ACCESS_WARN_ENV, "0") == "1"

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

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

        When the underlying warp array has dtype ``wp.quatf`` and the
        ``WARN_ON_TORCH_QUATF_ACCESS`` environment variable is set to ``"1"``,
        each read emits a :class:`UserWarning` pointing at the call site. This
        is a runtime aid for migrating Isaac Lab 2.x code (which used the
        ``(w, x, y, z)`` quaternion convention) to Isaac Lab 3.x's
        ``(x, y, z, w)`` convention.
        """
        if self._is_quatf and self._quatf_access_warning_enabled():
            warnings.warn(
                "Reading .torch on a wp.quatf-typed ProxyArray. The Isaac Lab"
                " quaternion convention changed from (w, x, y, z) in 2.x to"
                " (x, y, z, w) in 3.x. If your code assumes the old order,"
                " this is likely the source of incorrect rotations."
                f" Unset {_QUATF_ACCESS_WARN_ENV} to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
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
