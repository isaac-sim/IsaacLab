# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped restoration for backend-global settings used by in-process benchmarks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def scoped_attribute(target: Any, name: str, value: Any) -> Iterator[None]:
    """Temporarily assign an attribute and restore its previous value."""
    previous = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, previous)


@contextmanager
def scoped_torch_backend_flags(torch_module: Any) -> Iterator[None]:
    """Temporarily configure Torch backend flags used by RL training."""
    cuda_matmul = torch_module.backends.cuda.matmul
    cudnn = torch_module.backends.cudnn
    previous = (
        cuda_matmul.allow_tf32,
        cudnn.allow_tf32,
        cudnn.deterministic,
        cudnn.benchmark,
    )
    cuda_matmul.allow_tf32 = True
    cudnn.allow_tf32 = True
    cudnn.deterministic = False
    cudnn.benchmark = False
    try:
        yield
    finally:
        cuda_matmul.allow_tf32 = previous[0]
        cudnn.allow_tf32 = previous[1]
        cudnn.deterministic = previous[2]
        cudnn.benchmark = previous[3]
