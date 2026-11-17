# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selectors that run on Isaac Sim (Kit) warn about their deprecation and keep working."""

import warnings

import pytest

from isaaclab.utils.backend_utils import warn_kit_selector


@pytest.mark.parametrize(
    ("name", "replacement"),
    [("isaacsim_physx", "ovphysx"), ("isaacsim_rtx", "ovrtx")],
)
def test_isaac_sim_selectors_warn_with_their_replacement(name, replacement):
    with pytest.warns(FutureWarning, match=rf"'{name}' runs on Isaac Sim .*{replacement}"):
        warn_kit_selector(name)


def test_kit_free_selectors_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("ovphysx", "newton_mjwarp", "ovrtx", "newton_renderer"):
            warn_kit_selector(name)
