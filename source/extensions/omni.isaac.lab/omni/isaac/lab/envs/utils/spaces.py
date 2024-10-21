# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import numpy as np
import torch
from typing import Any

from ..common import SpaceType


def spec_to_gym_space(spec: SpaceType) -> gym.spaces.Space:
    """Generate an appropriate Gymnasium space according to the given space specification.

    Args:
        spec: Space specification.

    Returns:
        Gymnasium space.

    Raises:
        ValueError: If the given space specification is not valid/supported.
    """
    if isinstance(spec, gym.spaces.Space):
        return spec
    # fundamental spaces
    # Box
    elif isinstance(spec, int):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(spec,))
    elif isinstance(spec, list) and all(isinstance(x, int) for x in spec):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=spec)
    # Discrete
    elif isinstance(spec, set) and len(spec) == 1:
        return gym.spaces.Discrete(n=next(iter(spec)))
    # MultiDiscrete
    elif isinstance(spec, list) and all(isinstance(x, set) and len(x) == 1 for x in spec):
        return gym.spaces.MultiDiscrete(nvec=[next(iter(x)) for x in spec])
    # composite spaces
    # Tuple
    elif isinstance(spec, tuple):
        return gym.spaces.Tuple([spec_to_gym_space(x) for x in spec])
    # Dict
    elif isinstance(spec, dict):
        return gym.spaces.Dict({k: spec_to_gym_space(v) for k, v in spec.items()})
    raise ValueError(f"Unsupported space specification: {spec}")


def sample_space(space: gym.spaces.Space, device: str, batch_size: int = -1, fill_value: float | None = None) -> Any:
    """Sample a Gymnasium space where the data container are PyTorch tensors.

    Args:
        space: Gymnasium space.
        device: The device where the tensor should be created.
        batch_size: Batch size. If the specified value is greater than zero, a batched space will be created and sampled from it.
        fill_value: The value to fill the created tensors with. If None (default value), tensors will keep their random values.

    Returns:
        Tensorized sampled space.
    """

    def tensorize(s, x):
        if isinstance(s, gym.spaces.Box):
            tensor = torch.tensor(x, device=device, dtype=torch.float32).reshape(batch_size, *s.shape)
            if fill_value is not None:
                tensor.fill_(fill_value)
            return tensor
        elif isinstance(s, gym.spaces.Discrete):
            if isinstance(x, np.ndarray):
                tensor = torch.tensor(x, device=device, dtype=torch.int64).reshape(batch_size, 1)
                if fill_value is not None:
                    tensor.fill_(int(fill_value))
                return tensor
            elif isinstance(x, np.number) or type(x) in [int, float]:
                tensor = torch.tensor([x], device=device, dtype=torch.int64).reshape(batch_size, 1)
                if fill_value is not None:
                    tensor.fill_(int(fill_value))
                return tensor
        elif isinstance(s, gym.spaces.MultiDiscrete):
            if isinstance(x, np.ndarray):
                tensor = torch.tensor(x, device=device, dtype=torch.int64).reshape(batch_size, *s.shape)
                if fill_value is not None:
                    tensor.fill_(int(fill_value))
                return tensor
        elif isinstance(s, gym.spaces.Dict):
            return {k: tensorize(_s, x[k]) for k, _s in s.items()}
        elif isinstance(s, gym.spaces.Tuple):
            return tuple([tensorize(_s, v) for _s, v in zip(s, x)])

    sample = (gym.vector.utils.batch_space(space, batch_size) if batch_size > 0 else space).sample()
    return tensorize(space, sample)
