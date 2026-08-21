# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import namedtuple

import pytest

from isaaclab.utils.dict import replace_slices_with_strings, replace_strings_with_slices

pytestmark = pytest.mark.unit


def test_slice_round_trip_preserves_tuple_containers():
    """Slices nested in tuples should serialize and deserialize without changing container types."""
    data = {
        "selectors": (
            slice(None, 4, 2),
            [slice(1, None, None)],
            ("unchanged", slice(-3, -1, 1)),
        )
    }

    serialized = replace_slices_with_strings(data)

    assert isinstance(serialized["selectors"], tuple)
    assert isinstance(serialized["selectors"][1], list)
    assert isinstance(serialized["selectors"][2], tuple)
    assert serialized["selectors"][0] == "slice(None,4,2)"
    assert serialized["selectors"][1][0] == "slice(1,None,None)"
    assert serialized["selectors"][2][1] == "slice(-3,-1,1)"

    restored = replace_strings_with_slices(serialized)

    assert restored == data
    assert isinstance(restored["selectors"], tuple)
    assert isinstance(restored["selectors"][1], list)
    assert isinstance(restored["selectors"][2], tuple)


def test_slice_conversion_leaves_tuple_subclasses_unchanged():
    """Tuple subclasses that previously passed through should not be reconstructed through an incompatible constructor."""
    Pair = namedtuple("Pair", ["selector", "label"])
    data = {"pair": Pair(slice(1, 3), "value")}

    serialized = replace_slices_with_strings(data)
    restored = replace_strings_with_slices(serialized)

    assert serialized["pair"] is data["pair"]
    assert restored["pair"] is data["pair"]
