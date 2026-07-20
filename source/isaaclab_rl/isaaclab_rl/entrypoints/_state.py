# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped restoration for backend-global reinforcement learning settings."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import ModuleType

_MISSING = object()


@contextmanager
def preserve_attribute(target: object, name: str) -> Iterator[None]:
    """Restore an attribute after a scoped operation, including an initially missing attribute."""
    previous = getattr(target, name, _MISSING)
    try:
        yield
    finally:
        if previous is _MISSING:
            if hasattr(target, name):
                delattr(target, name)
        else:
            setattr(target, name, previous)


@contextmanager
def scoped_torch_backend_flags(torch_module: ModuleType) -> Iterator[None]:
    """Temporarily configure the Torch backend flags used by RSL-RL training."""
    cuda_matmul = torch_module.backends.cuda.matmul
    cudnn = torch_module.backends.cudnn
    settings = (
        (cuda_matmul, "allow_tf32", True),
        (cudnn, "allow_tf32", True),
        (cudnn, "deterministic", False),
        (cudnn, "benchmark", False),
    )
    with ExitStack() as cleanup:
        for target, name, value in settings:
            cleanup.enter_context(preserve_attribute(target, name))
            setattr(target, name, value)
        yield
