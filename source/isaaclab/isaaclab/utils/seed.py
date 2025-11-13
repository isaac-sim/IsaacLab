# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This method has been adjusted from
# https://github.com/isaac-sim/IsaacSim/blob/main/source/extensions/isaacsim.core.utils/python/impl/torch/maths.py

import numpy as np
import os
import random
import torch

import warp as wp


def configure_seed(seed: int | None, torch_deterministic: bool = False) -> int:
    """Set seed across all random number generators (torch, numpy, random, warp).

    Args:
        seed: The random seed value. If None, generates a random seed.
        torch_deterministic: If True, enables deterministic mode for torch operations.

    Returns:
        The seed value that was set.
    """
    if seed is None or seed == -1:
        if torch_deterministic:
            seed = 42
        else:
            seed = np.random.randint(0, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wp.rand_init(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
