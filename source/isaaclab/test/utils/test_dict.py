# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import enum
import random

import numpy as np
import pytest
import torch

import isaaclab.utils.dict as dict_utils
import isaaclab.utils.string as string_utils

pytestmark = pytest.mark.unit


def _test_function(x):
    """Test function for string <-> callable conversion."""
    return x**2


def _test_lambda_function(x):
    """Test function for string <-> callable conversion."""
    return x**2


def test_print_dict():
    """Test printing of dictionary."""
    # create a complex nested dictionary
    test_dict = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6}},
        "i": 7,
        "j": lambda x: x**2,  # noqa: E731
        "k": dict_utils.class_to_dict,
    }
    # print the dictionary
    dict_utils.print_dict(test_dict)


def test_string_callable_function_conversion():
    """Test string <-> callable conversion for function."""

    # convert function to string
    test_string = dict_utils.callable_to_string(_test_function)
    # convert string to function
    test_function_2 = string_utils.string_to_callable(test_string)
    # check that functions are the same
    assert _test_function(2) == test_function_2(2)


def test_string_callable_function_with_lambda_in_name_conversion():
    """Test string <-> callable conversion for function which has lambda in its name."""

    # convert function to string
    test_string = dict_utils.callable_to_string(_test_lambda_function)
    # convert string to function
    test_function_2 = string_utils.string_to_callable(test_string)
    # check that functions are the same
    assert _test_function(2) == test_function_2(2)


def test_string_callable_lambda_conversion():
    """Test string <-> callable conversion for lambda expression."""

    # create lambda function
    func = lambda x: x**2  # noqa: E731
    # convert function to string
    test_string = dict_utils.callable_to_string(func)
    # convert string to function
    func_2 = string_utils.string_to_callable(test_string)
    # check that functions are the same
    assert test_string == "lambda x: x**2"
    assert func(2) == func_2(2)


def test_dict_to_md5():
    """Test MD5 hash generation for dictionary."""
    # create a complex nested dictionary
    test_dict = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6}},
        "i": random.random(),
        "k": dict_utils.callable_to_string(dict_utils.class_to_dict),
    }
    # generate the MD5 hash
    md5_hash_1 = dict_utils.dict_to_md5_hash(test_dict)

    # check that the hash is correct even after multiple calls
    for _ in range(200):
        md5_hash_2 = dict_utils.dict_to_md5_hash(test_dict)
        assert md5_hash_1 == md5_hash_2


class _CallableCfg:
    class_type = _test_function


def test_update_class_from_dict_keeps_callable_string_lazy():
    """Callable-string updates should remain lazy via ResolvableString."""
    cfg = _CallableCfg()
    dict_utils.update_class_from_dict(cfg, {"class_type": "math:sin"})

    assert isinstance(cfg.class_type, string_utils.ResolvableString)
    # Dunder probing should not force resolution/import side effects.
    assert hasattr(cfg.class_type, "__dataclass_fields__") is False
    # Runtime use still resolves correctly.
    assert pytest.approx(cfg.class_type(0.0), rel=0.0, abs=1e-9) == 0.0


def test_update_class_from_dict_does_not_rewrap_resolvable_string():
    """Existing ResolvableString should be preserved, not re-wrapped."""
    cfg = _CallableCfg()
    existing = string_utils.ResolvableString("math:sin")
    dict_utils.update_class_from_dict(cfg, {"class_type": existing})

    assert cfg.class_type is existing


class _Flavor(enum.StrEnum):
    VANILLA = "vanilla"


class _Level(enum.IntEnum):
    LOW = 1


class _Flavors(enum.StrEnum):
    CHOCOLATE = "chocolate"
    STRAWBERRY = "strawberry"


class _EnumCfg:
    """Config holding enum members, which carry a ``__dict__`` of enum internals."""

    flavor: _Flavor = _Flavor.VANILLA
    level: _Level = _Level.LOW
    scoops: list[_Flavors] = [_Flavors.CHOCOLATE]
    cone: tuple[_Flavors, ...] = (_Flavors.STRAWBERRY,)

    def __init__(self):
        self.flavor = _Flavor.VANILLA
        self.level = _Level.LOW
        self.scoops = [_Flavors.CHOCOLATE]
        self.cone = (_Flavors.STRAWBERRY,)


def test_class_to_dict_serializes_enums_as_values():
    """Enum members should serialize to the value they stand for, not to their internals."""
    data = dict_utils.class_to_dict(_EnumCfg())

    assert data["flavor"] == "vanilla"
    assert data["level"] == 1
    # the enum internals must not leak into the output
    assert not isinstance(data["flavor"], dict)
    assert not isinstance(data["level"], dict)


def test_update_class_from_dict_restores_enums_from_values():
    """Serializing and reloading a config should hand back enum members, not raw scalars."""
    cfg = _EnumCfg()

    dict_utils.update_class_from_dict(cfg, dict_utils.class_to_dict(_EnumCfg()))

    assert cfg.flavor is _Flavor.VANILLA
    assert cfg.level is _Level.LOW
    # members inside a list or tuple survive the round trip too: the flat-iterable path
    # replaces the container wholesale, so it has to rebuild them itself
    assert cfg.scoops == [_Flavors.CHOCOLATE]
    assert all(isinstance(el, _Flavors) for el in cfg.scoops)
    assert cfg.cone == (_Flavors.STRAWBERRY,)
    assert all(isinstance(el, _Flavors) for el in cfg.cone)


def test_nested_conversion_preserves_requested_backend():
    """Nested conversions should preserve the backend selected by the caller."""
    data = {"outer": {"values": np.array([1.0, 2.0, 3.0], dtype=np.float32)}}

    converted = dict_utils.convert_dict_to_backend(data, backend="torch", array_types=("numpy",))

    assert isinstance(converted["outer"]["values"], torch.Tensor)
    torch.testing.assert_close(converted["outer"]["values"], torch.tensor([1.0, 2.0, 3.0]))
