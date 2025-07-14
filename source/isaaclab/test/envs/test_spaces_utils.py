# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import torch
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

from isaaclab.envs.utils.spaces import deserialize_space, sample_space, serialize_space, spec_to_gym_space


def test_spec_to_gym_space():
    """Test conversion of specs to gym spaces."""
    # fundamental spaces
    # Box
    space = spec_to_gym_space(1)
    assert isinstance(space, Box)
    assert space.shape == (1,)
    space = spec_to_gym_space([1, 2, 3, 4, 5])
    assert isinstance(space, Box)
    assert space.shape == (1, 2, 3, 4, 5)
    space = spec_to_gym_space(Box(low=-1.0, high=1.0, shape=(1, 2)))
    assert isinstance(space, Box)
    # Discrete
    space = spec_to_gym_space({2})
    assert isinstance(space, Discrete)
    assert space.n == 2
    space = spec_to_gym_space(Discrete(2))
    assert isinstance(space, Discrete)
    # MultiDiscrete
    space = spec_to_gym_space([{1}, {2}, {3}])
    assert isinstance(space, MultiDiscrete)
    assert space.nvec.shape == (3,)
    space = spec_to_gym_space(MultiDiscrete(np.array([1, 2, 3])))
    assert isinstance(space, MultiDiscrete)
    # composite spaces
    # Tuple
    space = spec_to_gym_space(([1, 2, 3, 4, 5], {2}, [{1}, {2}, {3}]))
    assert isinstance(space, Tuple)
    assert len(space) == 3
    assert isinstance(space[0], Box)
    assert isinstance(space[1], Discrete)
    assert isinstance(space[2], MultiDiscrete)
    space = spec_to_gym_space(Tuple((Box(-1, 1, shape=(1,)), Discrete(2))))
    assert isinstance(space, Tuple)
    # Dict
    space = spec_to_gym_space({"box": [1, 2, 3, 4, 5], "discrete": {2}, "multi_discrete": [{1}, {2}, {3}]})
    assert isinstance(space, Dict)
    assert len(space) == 3
    assert isinstance(space["box"], Box)
    assert isinstance(space["discrete"], Discrete)
    assert isinstance(space["multi_discrete"], MultiDiscrete)
    space = spec_to_gym_space(Dict({"box": Box(-1, 1, shape=(1,)), "discrete": Discrete(2)}))
    assert isinstance(space, Dict)


def test_sample_space():
    """Test sampling from gym spaces."""
    device = "cpu"
    # fundamental spaces
    # Box
    sample = sample_space(Box(low=-1.0, high=1.0, shape=(1, 2)), device, batch_size=1)
    assert isinstance(sample, torch.Tensor)
    _check_tensorized(sample, batch_size=1)
    # Discrete
    sample = sample_space(Discrete(2), device, batch_size=2)
    assert isinstance(sample, torch.Tensor)
    _check_tensorized(sample, batch_size=2)
    # MultiDiscrete
    sample = sample_space(MultiDiscrete(np.array([1, 2, 3])), device, batch_size=3)
    assert isinstance(sample, torch.Tensor)
    _check_tensorized(sample, batch_size=3)
    # composite spaces
    # Tuple
    sample = sample_space(Tuple((Box(-1, 1, shape=(1,)), Discrete(2))), device, batch_size=4)
    assert isinstance(sample, (tuple, list))
    _check_tensorized(sample, batch_size=4)
    # Dict
    sample = sample_space(Dict({"box": Box(-1, 1, shape=(1,)), "discrete": Discrete(2)}), device, batch_size=5)
    assert isinstance(sample, dict)
    _check_tensorized(sample, batch_size=5)


def test_space_serialization_deserialization():
    """Test serialization and deserialization of gym spaces."""
    # fundamental spaces
    # Box
    space = 1
    output = deserialize_space(serialize_space(space))
    assert space == output
    space = [1, 2, 3, 4, 5]
    output = deserialize_space(serialize_space(space))
    assert space == output
    space = Box(low=-1.0, high=1.0, shape=(1, 2))
    output = deserialize_space(serialize_space(space))
    assert isinstance(output, Box)
    assert (space.low == output.low).all()
    assert (space.high == output.high).all()
    assert space.shape == output.shape
    # Discrete
    space = {2}
    output = deserialize_space(serialize_space(space))
    assert space == output
    space = Discrete(2)
    output = deserialize_space(serialize_space(space))
    assert isinstance(output, Discrete)
    assert space.n == output.n
    # MultiDiscrete
    space = [{1}, {2}, {3}]
    output = deserialize_space(serialize_space(space))
    assert space == output
    space = MultiDiscrete(np.array([1, 2, 3]))
    output = deserialize_space(serialize_space(space))
    assert isinstance(output, MultiDiscrete)
    assert (space.nvec == output.nvec).all()
    # composite spaces
    # Tuple
    space = ([1, 2, 3, 4, 5], {2}, [{1}, {2}, {3}])
    output = deserialize_space(serialize_space(space))
    assert space == output
    space = Tuple((Box(-1, 1, shape=(1,)), Discrete(2)))
    output = deserialize_space(serialize_space(space))
    assert isinstance(output, Tuple)
    assert len(output) == 2
    assert isinstance(output[0], Box)
    assert isinstance(output[1], Discrete)
    # Dict
    space = {"box": [1, 2, 3, 4, 5], "discrete": {2}, "multi_discrete": [{1}, {2}, {3}]}
    output = deserialize_space(serialize_space(space))
    assert space == output
    space = Dict({"box": Box(-1, 1, shape=(1,)), "discrete": Discrete(2)})
    output = deserialize_space(serialize_space(space))
    assert isinstance(output, Dict)
    assert len(output) == 2
    assert isinstance(output["box"], Box)
    assert isinstance(output["discrete"], Discrete)


def _check_tensorized(sample, batch_size):
    """Helper function to check if a sample is properly tensorized."""
    if isinstance(sample, (tuple, list)):
        list(map(_check_tensorized, sample, [batch_size] * len(sample)))
    elif isinstance(sample, dict):
        list(map(_check_tensorized, sample.values(), [batch_size] * len(sample)))
    else:
        assert isinstance(sample, torch.Tensor)
        assert sample.shape[0] == batch_size
