# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from ..warp import ProxyArray
    from .tensor_processor_cfg import TensorProcessorCfg


TensorProcessorData = Union[  # noqa: UP007
    torch.Tensor,
    "ProxyArray",
    list["TensorProcessorData"],
    tuple["TensorProcessorData", ...],
    dict[str, "TensorProcessorData"],
]
"""A tensor or a nested collection of tensors accepted or returned by a processor."""


class TensorProcessor(ABC):
    """Transform tensors or nested collections, optionally retaining state between calls.

    Subclasses implement ``__call__`` and define the input and output structures they support;
    these need not match. Collections may contain tensors, ProxyArrays, lists, tuples, or
    dictionaries. Inputs are borrowed read-only. Stateful processors override ``reset``;
    processors that own external resources override ``close``.

    Observation processors accept and return one tensor of the same shape. Their configuration,
    dimensions, and device are supplied by the observation manager. Structured processors may
    instead define a constructor accepting input buffers and call ``super().__init__(cfg)``.
    Constructors inspect input metadata and allocate outputs without reading input values or
    advancing state. Persistent outputs remain valid until the next call; consumers retaining
    results must copy them. Warp-based GPU work uses the caller's current Warp stream.
    """

    def __init__(
        self, cfg: TensorProcessorCfg, data_dim: tuple[int, ...] | None = None, device: str | None = None
    ) -> None:
        """Initializes the processor class.

        Args:
            cfg: Configuration parameters.
            data_dim: Dimensions for a single-tensor processor. The first element is the batch size,
                usually the number of environments. Omit when the subclass handles structured inputs.
            device: Device for a single-tensor processor. Structured processors may infer it from inputs.
        """
        self._cfg = cfg
        self._data_dim = data_dim
        self._device = device

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset state for selected batch entries, or all entries when None.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        pass

    def close(self) -> None:
        """Release owned resources; repeated calls must have no effect."""
        pass

    @abstractmethod
    def __call__(self, data: TensorProcessorData) -> TensorProcessorData:
        """Process inputs without mutating them and return the result.

        Args:
            data: Tensor or nested collection supported by the implementation.

        Returns:
            Processed data with the structure defined by the implementation.
        """
        raise NotImplementedError
