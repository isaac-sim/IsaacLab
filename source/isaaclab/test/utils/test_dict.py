# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import enum

import numpy as np
import pytest
import torch
import warp as wp

import isaaclab.utils.dict as dict_utils
import isaaclab.utils.string as string_utils

pytestmark = pytest.mark.unit


def _square(x):
    return x**2


def _lambda_named_function(x):
    """Function whose name contains 'lambda' and must not be mistaken for a lambda expression."""
    return x**2


def test_print_dict_handles_nested_values_and_callables(capsys):
    dict_utils.print_dict({"a": 1, "c": {"d": 3, "f": {"g": 5}}, "j": lambda x: x**2, "k": dict_utils.class_to_dict})
    output = capsys.readouterr().out
    assert "g" in output and "class_to_dict" in output


def test_callable_string_round_trip():
    for func in (_square, _lambda_named_function):
        as_string = dict_utils.callable_to_string(func)
        assert as_string == f"{__name__}:{func.__name__}"
        assert string_utils.string_to_callable(as_string) is func

    square = lambda x: x**2  # noqa: E731
    as_string = dict_utils.callable_to_string(square)
    assert as_string == "lambda x: x**2"
    assert string_utils.string_to_callable(as_string)(3) == 9


def test_dict_to_md5_hash_is_deterministic():
    data = {"a": 1, "c": {"d": 3, "f": {"g": 5}}, "i": 0.12345, "k": dict_utils.callable_to_string(_square)}
    assert dict_utils.dict_to_md5_hash(data) == dict_utils.dict_to_md5_hash(dict(data))
    assert dict_utils.dict_to_md5_hash(data) != dict_utils.dict_to_md5_hash({**data, "i": 0.5})


class _CallableCfg:
    class_type = _square


def test_update_class_from_dict_keeps_callable_strings_lazy():
    cfg = _CallableCfg()
    dict_utils.update_class_from_dict(cfg, {"class_type": "math:sin"})
    assert isinstance(cfg.class_type, string_utils.ResolvableString)
    assert hasattr(cfg.class_type, "__dataclass_fields__") is False  # dunder probing does not resolve
    assert cfg.class_type(0.0) == pytest.approx(0.0)

    existing = string_utils.ResolvableString("math:sin")
    dict_utils.update_class_from_dict(cfg, {"class_type": existing})
    assert cfg.class_type is existing  # not re-wrapped


class _Flavor(enum.StrEnum):
    VANILLA = "vanilla"
    CHOCOLATE = "chocolate"


class _Level(enum.IntEnum):
    LOW = 1


class _EnumCfg:
    def __init__(self):
        self.flavor = _Flavor.VANILLA
        self.level = _Level.LOW
        self.scoops = [_Flavor.CHOCOLATE]
        self.cone = (_Flavor.CHOCOLATE,)


def test_enum_members_round_trip_through_dict():
    data = dict_utils.class_to_dict(_EnumCfg())
    assert data["flavor"] == "vanilla" and data["level"] == 1

    cfg = _EnumCfg()
    dict_utils.update_class_from_dict(cfg, data)
    assert cfg.flavor is _Flavor.VANILLA and cfg.level is _Level.LOW
    # containers are replaced wholesale, so their elements must be rebuilt as enum members too
    assert cfg.scoops == [_Flavor.CHOCOLATE] and isinstance(cfg.scoops[0], _Flavor)
    assert cfg.cone == (_Flavor.CHOCOLATE,) and isinstance(cfg.cone[0], _Flavor)


@pytest.mark.parametrize(("backend", "array_type"), [("torch", torch.Tensor), ("warp", wp.array)])
def test_convert_dict_to_backend_recurses(backend, array_type):
    data = {"outer": {"values": np.array([1.0, 2.0, 3.0], dtype=np.float32)}}
    converted = dict_utils.convert_dict_to_backend(data, backend=backend, array_types=("numpy",))
    values = converted["outer"]["values"]
    assert isinstance(values, array_type)
    np.testing.assert_array_equal(values.numpy(), data["outer"]["values"])
