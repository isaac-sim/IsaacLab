# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import random
import weakref
from collections.abc import Callable

import numpy as np
import torch
import warp as wp

_seed_listeners: list[weakref.WeakMethod] = []


def add_seed_listener(listener: Callable[[int], None]) -> None:
    """Call a bound method with the seed after every :func:`configure_seed`.

    Args:
        listener: Bound method that reseeds its object's random state. The object is held weakly, so the listener
            stops when the object is deleted.
    """
    _seed_listeners.append(weakref.WeakMethod(listener))


def configure_seed(seed: int | None, torch_deterministic: bool = False) -> int:
    """Set seed across all random number generators (torch, numpy, random, warp) and pass it to the seed listeners.

    Args:
        seed: The random seed value. If None, generates a random seed.
        torch_deterministic: Whether torch operations should use deterministic mode. ``True`` enables PyTorch
            deterministic algorithms; ``False`` actively disables them, including a mode enabled by an earlier call.

    Returns:
        The seed value that was set.
    """
    if seed is None or seed == -1:
        seed = 42 if torch_deterministic else random.randint(0, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wp.rand_init(seed)
    for listener_ref in list(_seed_listeners):
        if (listener := listener_ref()) is None:
            _seed_listeners.remove(listener_ref)
        else:
            listener(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

    return seed
