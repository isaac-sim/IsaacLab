# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter

import pytest

from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString, callable_to_string, string_to_callable

pytestmark = pytest.mark.unit


@configclass
class _NestedCallableCfg:
    updater: str = "collections:Counter.update"


def test_configclass_wraps_nested_callable_reference():
    cfg = _NestedCallableCfg()

    assert isinstance(cfg.updater, ResolvableString)
    counter = Counter()
    cfg.updater(counter, {"value": 2})
    assert counter["value"] == 2


def test_nested_callable_reference_round_trips():
    reference = callable_to_string(Counter.update)

    assert reference == "collections:Counter.update"
    assert string_to_callable(reference) is Counter.update

    counter = Counter()
    ResolvableString(reference)(counter, {"value": 2})
    assert counter["value"] == 2


def test_missing_nested_callable_attribute_raises_value_error():
    with pytest.raises(ValueError, match="Could not resolve"):
        string_to_callable("collections:Counter.not_a_method")


def _make_exported_local_callable():
    def exported_local_callable():
        return "ok"

    return exported_local_callable


exported_local_callable = _make_exported_local_callable()


def test_local_callable_serialization_falls_back_to_resolvable_name():
    reference = callable_to_string(exported_local_callable)

    assert reference == f"{__name__}:exported_local_callable"
    assert string_to_callable(reference) is exported_local_callable


def test_instance_bound_method_preserves_legacy_simple_name():
    counter = Counter()

    assert callable_to_string(counter.update) == "collections:update"
