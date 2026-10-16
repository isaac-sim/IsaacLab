# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..warp import ProxyArray

if TYPE_CHECKING:
    from .dict_tensor_processor_cfg import DictTensorProcessorCfg


class DictTensorProcessor(ABC):
    """Transform a dictionary of named tensors into a dictionary of named tensors.

    This is the dictionary counterpart of :class:`TensorProcessor`. Tensor values use
    :class:`ProxyArray` so processors can use Warp or Torch without copying. Inputs are
    borrowed read-only. Constructors inspect their shape, dtype and device and allocate
    persistent outputs without reading input values or advancing processor state.

    Subclasses populate ``outputs`` with the buffers they produce, implement ``__call__``,
    and override ``reset`` or ``close`` when they own state or external resources. Calls
    return those same output buffers; values remain valid until the next call. Consumers
    that retain results must copy them. GPU work uses the caller's current Warp stream.
    """

    def __init__(self, cfg: DictTensorProcessorCfg, data: dict[str, ProxyArray]) -> None:
        self.cfg = cfg
        self.outputs: dict[str, ProxyArray] = {}

    @abstractmethod
    def __call__(self, data: dict[str, ProxyArray]) -> dict[str, ProxyArray]:
        """Process current inputs and return the named outputs without mutating inputs."""
        raise NotImplementedError

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset state for selected batch entries, or all entries when None."""
        pass

    def close(self) -> None:
        """Release owned resources; repeated calls must have no effect."""
        pass
