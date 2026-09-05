# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`isaaclab_tasks.utils.parse_cfg.parse_env_cfg`."""

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def test_parse_env_cfg_rejects_bare_string_overrides():
    """A bare string is itself a ``Sequence[str]`` of characters; catch it explicitly."""
    with pytest.raises(TypeError, match="bare string"):
        parse_env_cfg("Isaac-Cartpole", overrides="physics=isaacsim_physx")


def test_parse_env_cfg_accepts_list_overrides():
    """A properly wrapped override list should apply without error."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", overrides=["physics=isaacsim_physx"])
    assert env_cfg is not None
