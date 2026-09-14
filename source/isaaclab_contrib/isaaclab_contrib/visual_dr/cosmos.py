# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos construction boundary; tensor-native model adaptation is the next slice."""

from collections.abc import Callable
from dataclasses import dataclass

from .runtime import DRBackend


@dataclass(frozen=True)
class CosmosDRCfg:
    """Model-worker settings; precision and compilation are capabilities to validate."""

    checkpoint: str
    prompt: str
    fp8: bool = False
    compile: bool = False


def create_cosmos_backend(cfg: CosmosDRCfg, factory: Callable[[CosmosDRCfg], DRBackend] | None = None) -> DRBackend:
    """Build an application-supplied CUDA adapter without importing any trainer.

    The factory owns model loading, mask-guided sampling, FP8, compile and paging.
    There is deliberately no file/NumPy inference fallback for live observations.
    """
    if factory is None:
        raise NotImplementedError(
            "Cosmos needs a tensor-native DRBackend adapter; the file-oriented inference API is not sufficient. "
            "Provide a CUDA adapter factory for the patched Cosmos source checkout."
        )
    return factory(cfg)
