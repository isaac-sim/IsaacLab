# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for MDP event selector classification."""

import pytest

from isaaclab.envs.mdp.events import _is_all_body_selection


@pytest.mark.parametrize(
    ("body_ids", "num_bodies", "expected"),
    [
        (slice(None), 3, True),
        ([0, 1, 2], 3, True),
        ([2, 0, 1], 3, True),
        ([0, 1], 3, False),
        ([0, 1, 1], 3, False),
    ],
)
def test_is_all_body_selection_classifies_complete_and_partial_selectors(body_ids, num_bodies, expected):
    """Only the sentinel and explicit complete body selections should classify as whole-asset."""
    assert _is_all_body_selection(body_ids, num_bodies) is expected
