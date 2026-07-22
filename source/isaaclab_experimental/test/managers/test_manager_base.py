# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp manager reset-mask contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import warp as wp
from isaaclab_experimental.managers.action_manager import ActionManager
from isaaclab_experimental.managers.manager_base import _resolve_reset_mask
from isaaclab_experimental.utils.warp import WarpCapturable


def test_reset_mask_rejects_wrong_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manager kernels should fail before receiving a mask from another device."""
    owner = SimpleNamespace(num_envs=2, device="expected")
    env_mask = wp.zeros(2, dtype=wp.bool, device="cpu")
    expected_device = SimpleNamespace(alias="expected")
    monkeypatch.setattr(wp, "get_device", lambda device: expected_device)

    with pytest.raises(ValueError, match="env_mask must be on"):
        _resolve_reset_mask(owner, None, env_mask)


def test_class_term_capturability_is_registered() -> None:
    """Class-based manager terms should honor explicit capture metadata."""

    @WarpCapturable(False)
    class HostTerm:
        pass

    graph_cache = Mock()
    manager = object.__new__(ActionManager)
    manager._env = SimpleNamespace(_warp_graph_cache=graph_cache)

    manager._register_term_capturability(HostTerm)

    graph_cache.register_capturability.assert_called_once_with("ActionManager", False)
