# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp observation manager configuration parsing."""

from types import SimpleNamespace

import warp as wp
from isaaclab_experimental.managers import ObservationManager

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


def constant_obs(env, out: wp.array, out_dim: int):
    """Warp-first observation term; the manager reads its width from ``out_dim``."""
    out.fill_(1.0)


@configclass
class PolicyCfg(ObservationGroupCfg):
    """Group with two terms and every group-level setting at its default."""

    first = ObservationTermCfg(func=constant_obs, params={"out_dim": 2})
    second = ObservationTermCfg(func=constant_obs, params={"out_dim": 3})


def test_group_settings_are_not_parsed_as_terms():
    """Every :class:`ObservationGroupCfg` field, including ``history_order``, is a group setting, not a term."""
    env = SimpleNamespace(num_envs=4, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))

    manager = ObservationManager({"policy": PolicyCfg()}, env)

    assert manager.active_terms["policy"] == ["first", "second"]
    assert manager.group_obs_dim["policy"] == (5,)
