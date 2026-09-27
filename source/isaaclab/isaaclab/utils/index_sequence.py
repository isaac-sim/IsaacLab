# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolved integer selections with host access and cached Torch indexing."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from typing import Any, overload

import torch


class IndexSequence(Sequence[int]):
    """Read-only integer sequence with a device tensor created at construction.

    Python iteration, comparison, and scalar indexing use the host values. Torch
    operations, including ``data[:, indices]``, dispatch to the cached tensor through
    ``__torch_function__``. Tensor constructors do not use that protocol: use
    :attr:`torch` or :func:`isaaclab.utils.convert_to_torch` instead.

    Treat the returned tensor as read-only. To change a selection, construct a new
    sequence. Slicing this sequence returns a Python list; slicing its Torch view
    returns a tensor view. Neither operation reads indices back from the device.
    """

    __slots__ = ("_values", "_torch")

    def __init__(self, indices: Sequence[int], device: str | torch.device):
        """Store the host indices and upload their integer tensor once.

        Args:
            indices: Ordered integer indices, including duplicates and negative indices.
            device: Device used for Torch indexing.
        """
        self._values = tuple(map(operator.index, indices))
        self._torch = torch.tensor(self._values, dtype=torch.long, device=device)

    @property
    def torch(self) -> torch.Tensor:
        """Cached integer tensor. Callers must not modify its contents."""
        return self._torch

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[int]:
        return iter(self._values)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index: int | slice) -> int | list[int]:
        value = self._values[index]
        return list(value) if isinstance(index, slice) else value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return self._values == tuple(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"IndexSequence({list(self._values)!r}, device={str(self._torch.device)!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> IndexSequence:
        # Selections are read-only; configuration copies can share their device storage.
        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Unwrap selections, including selectors nested in indexing tuples."""

        def unwrap(value):
            if isinstance(value, cls):
                return value.torch
            if isinstance(value, (list, tuple)):
                return type(value)(unwrap(item) for item in value)
            return value

        return func(*unwrap(args), **{key: unwrap(value) for key, value in (kwargs or {}).items()})
