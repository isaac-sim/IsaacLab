# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import json
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


def serialize_space(space: SpaceType) -> str:
    """Serialize a space specification as JSON.

    Args:
        space: Space specification.

    Returns:
        Serialized JSON representation.
    """
    # Gymnasium spaces
    if isinstance(space, gym.spaces.Discrete):
        return json.dumps({"type": "gymnasium", "space": "Discrete", "n": int(space.n)})
    elif isinstance(space, gym.spaces.Box):
        return json.dumps({
            "type": "gymnasium",
            "space": "Box",
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": space.shape,
        })
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return json.dumps({"type": "gymnasium", "space": "MultiDiscrete", "nvec": space.nvec.tolist()})
    elif isinstance(space, gym.spaces.Tuple):
        return json.dumps({"type": "gymnasium", "space": "Tuple", "spaces": tuple(map(serialize_space, space.spaces))})
    elif isinstance(space, gym.spaces.Dict):
        return json.dumps(
            {"type": "gymnasium", "space": "Dict", "spaces": {k: serialize_space(v) for k, v in space.spaces.items()}}
        )
    # Python data types
    # Box
    elif isinstance(space, int) or (isinstance(space, list) and all(isinstance(x, int) for x in space)):
        return json.dumps({"type": "python", "space": "Box", "value": space})
    # Discrete
    elif isinstance(space, set) and len(space) == 1:
        return json.dumps({"type": "python", "space": "Discrete", "value": next(iter(space))})
    # MultiDiscrete
    elif isinstance(space, list) and all(isinstance(x, set) and len(x) == 1 for x in space):
        return json.dumps({"type": "python", "space": "MultiDiscrete", "value": [next(iter(x)) for x in space]})
    # composite spaces
    # Tuple
    elif isinstance(space, tuple):
        return json.dumps({"type": "python", "space": "Tuple", "value": [serialize_space(x) for x in space]})
    # Dict
    elif isinstance(space, dict):
        return json.dumps(
            {"type": "python", "space": "Dict", "value": {k: serialize_space(v) for k, v in space.items()}}
        )
    raise ValueError(f"Unsupported space ({space})")


def deserialize_space(string: str) -> gym.spaces.Space:
    """Deserialize a space specification encoded as JSON.

    Args:
        string: Serialized JSON representation.

    Returns:
        Space specification.
    """
    obj = json.loads(string)
    # Gymnasium spaces
    if obj["type"] == "gymnasium":
        if obj["space"] == "Discrete":
            return gym.spaces.Discrete(n=obj["n"])
        elif obj["space"] == "Box":
            return gym.spaces.Box(low=np.array(obj["low"]), high=np.array(obj["high"]), shape=obj["shape"])
        elif obj["space"] == "MultiDiscrete":
            return gym.spaces.MultiDiscrete(nvec=np.array(obj["nvec"]))
        elif obj["space"] == "Tuple":
            return gym.spaces.Tuple(spaces=tuple(map(deserialize_space, obj["spaces"])))
        elif obj["space"] == "Dict":
            return gym.spaces.Dict(spaces={k: deserialize_space(v) for k, v in obj["spaces"].items()})
        else:
            raise ValueError(f"Unsupported space ({obj['spaces']})")
    # Python data types
    elif obj["type"] == "python":
        if obj["space"] == "Discrete":
            return {obj["value"]}
        elif obj["space"] == "Box":
            return obj["value"]
        elif obj["space"] == "MultiDiscrete":
            return [{x} for x in obj["value"]]
        elif obj["space"] == "Tuple":
            return tuple(map(deserialize_space, obj["value"]))
        elif obj["space"] == "Dict":
            return {k: deserialize_space(v) for k, v in obj["value"].items()}
        else:
            raise ValueError(f"Unsupported space ({obj['spaces']})")
    else:
        raise ValueError(f"Unsupported type ({obj['type']})")


def replace_env_cfg_spaces_with_strings(env_cfg: object) -> object:
    """Replace spaces objects with their serialized JSON representations in an environment config.

    Args:
        env_cfg: Environment config instance.

    Returns:
        Environment config instance with spaces replaced if any.
    """
    for attr in ["observation_space", "action_space", "state_space"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, serialize_space(getattr(env_cfg, attr)))
    for attr in ["observation_spaces", "action_spaces"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, {k: serialize_space(v) for k, v in getattr(env_cfg, attr).items()})
    return env_cfg


def replace_strings_with_env_cfg_spaces(env_cfg: object) -> object:
    """Replace spaces objects with their serialized JSON representations in an environment config.

    Args:
        env_cfg: Environment config instance.

    Returns:
        Environment config instance with spaces replaced if any.
    """
    for attr in ["observation_space", "action_space", "state_space"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, deserialize_space(getattr(env_cfg, attr)))
    for attr in ["observation_spaces", "action_spaces"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, {k: deserialize_space(v) for k, v in getattr(env_cfg, attr).items()})
    return env_cfg
