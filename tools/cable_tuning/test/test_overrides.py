# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the override-application helper."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tools.cable_tuning.overrides import apply_overrides, parse_override_value


@dataclass
class _Inner:
    a: int = 1
    b: float = 2.0


@dataclass
class _Outer:
    inner: _Inner = field(default_factory=_Inner)
    name: str = "x"


def test_apply_overrides_scalar_path() -> None:
    obj = _Outer()
    apply_overrides(obj, {"inner.a": 42})
    assert obj.inner.a == 42


def test_apply_overrides_top_level() -> None:
    obj = _Outer()
    apply_overrides(obj, {"name": "y"})
    assert obj.name == "y"


def test_apply_overrides_unknown_path_raises() -> None:
    obj = _Outer()
    with pytest.raises(AttributeError):
        apply_overrides(obj, {"inner.does_not_exist": 0})


def test_parse_override_value_numeric() -> None:
    assert parse_override_value("1e5") == 1e5
    assert parse_override_value("40") == 40
    assert isinstance(parse_override_value("40"), int)
    assert parse_override_value("1.5") == 1.5


def test_parse_override_value_bool() -> None:
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False


def test_parse_override_value_string_fallback() -> None:
    assert parse_override_value("elliptic") == "elliptic"
