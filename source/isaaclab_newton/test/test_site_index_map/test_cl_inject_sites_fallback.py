# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for NewtonManager._cl_inject_sites_fallback site map structure."""

import pytest
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager


class MockBuilder:
    """Minimal stand-in for ModelBuilder."""

    def __init__(self, body_labels: list[str]):
        self.body_label = body_labels
        self._next_idx = 0

    def add_site(self, body: int, xform: wp.transform, label: str) -> int:
        idx = self._next_idx
        self._next_idx += 1
        return idx


class TestFallbackGlobalSite:
    """Global site (body_pattern=None) must produce a (int, None) entry."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder(["body0", "body1"])

    def test_global_site_entry_is_int_none_tuple(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {(None, tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert isinstance(global_idx, int)
        assert per_world is None

    def test_global_site_pending_cleared(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {(None, tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        assert len(NewtonManager._cl_pending_sites) == 0


class TestFallbackLocalSingleBody:
    """Single-body local site must produce a (None, [[idx]]) entry — one world."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder(["Robot/base", "Robot/hand"])

    def test_single_body_entry_shape(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/base", tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert global_idx is None
        assert isinstance(per_world, list)
        assert len(per_world) == 1  # one world
        assert len(per_world[0]) == 1  # one match
        assert isinstance(per_world[0][0], int)


class TestFallbackLocalWildcard:
    """Wildcard local site matching N bodies must produce (None, [[idx0..idxN-1]]) — one world."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder(["Robot/FL_foot", "Robot/FR_foot", "Robot/RL_foot", "Robot/RR_foot"])

    def test_wildcard_entry_shape(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/.*_foot", tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert global_idx is None
        assert len(per_world) == 1  # one world
        assert len(per_world[0]) == 4  # four bodies matched

    def test_no_match_raises(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/nonexistent", tuple(xform)): ("ft_0", xform)}
        with pytest.raises(ValueError):
            NewtonManager._cl_inject_sites_fallback()
