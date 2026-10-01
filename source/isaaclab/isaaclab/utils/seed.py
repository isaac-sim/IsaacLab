# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import random

import numpy as np
import torch
import warp as wp


class WarpRng:
    """Process-wide per-environment random number generator state for Warp kernels.

    Warp has no global generator: :func:`warp.rand_init` hashes a seed and an offset into a state that lives only
    for one kernel launch. Kernels that draw random numbers per environment therefore read ``state[env]``, draw, and
    store the advanced state back, so the next launch or CUDA-graph replay draws new numbers. This class holds that
    array once per process, so environments, managers and sensors share it instead of each allocating a copy.

    :meth:`initialize` must be called once the number of environments and the device are known; environments do this
    at construction, so users read :attr:`state` directly. :meth:`seed` is optional; without it, the state is seeded
    from :func:`torch.initial_seed`. Only kernels launched over environments, one thread per environment, may advance
    the state.
    """

    state: wp.array | None = None
    """The per-environment random number generator state, or None before :meth:`initialize`. Shape is (num_envs,)."""

    _seed: int | None = None

    @classmethod
    def seed(cls, seed: int) -> None:
        """Set the seed, and reseed the state in place if it is initialized.

        Args:
            seed: The random seed value.
        """
        cls._seed = seed
        if cls.state is not None:
            cls._reset_state()

    @classmethod
    def initialize(cls, num_envs: int, device: str) -> None:
        """Allocate and seed a new state. Environments call it once at construction.

        Args:
            num_envs: The number of environments.
            device: The device of the state.
        """
        cls.state = wp.empty(num_envs, dtype=wp.uint32, device=device)
        cls._reset_state()

    @classmethod
    def _reset_state(cls) -> None:
        """Seed every environment's state from the set seed, or from torch's initial seed without one."""
        from .warp.kernels import initialize_rng_state

        seed = cls._seed if cls._seed is not None else torch.initial_seed()
        wp.launch(
            initialize_rng_state,
            dim=cls.state.shape[0],
            inputs=[seed % 2**31],
            outputs=[cls.state],
            device=cls.state.device,
        )


def configure_seed(seed: int | None, torch_deterministic: bool = False) -> int:
    """Set seed across all random number generators (torch, numpy, random, warp).

    Args:
        seed: The random seed value. If None, generates a random seed.
        torch_deterministic: If True, enables deterministic mode for torch operations.

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
    WarpRng.seed(seed)

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
