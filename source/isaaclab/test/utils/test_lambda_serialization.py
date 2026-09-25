# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab.utils.string import callable_to_string, string_to_callable

pytestmark = pytest.mark.unit


def _make_tuple_lambda():
    return lambda x: (x, x + 1)


def _make_list_lambda():
    return lambda x: [x, x + 1, x + 2]  # trailing comments are not part of the lambda expression


def _make_unsafe_lambda():
    return lambda x: x.__class__


@pytest.mark.parametrize(
    ("factory", "expected_source", "argument", "expected_result"),
    [
        (_make_tuple_lambda, "lambda x: (x, x + 1)", 3, (3, 4)),
        (_make_list_lambda, "lambda x: [x, x + 1, x + 2]", 3, [3, 4, 5]),
    ],
)
def test_callable_to_string_preserves_commas_and_round_trips(factory, expected_source, argument, expected_result):
    serialized = callable_to_string(factory())

    assert serialized == expected_source
    assert string_to_callable(serialized)(argument) == expected_result


def test_serialized_lambda_still_uses_safe_resolution_checks():
    serialized = callable_to_string(_make_unsafe_lambda())

    with pytest.raises(ValueError, match="Unsafe lambda expression"):
        string_to_callable(serialized)
