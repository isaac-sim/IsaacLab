# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp manager reset-mask contracts."""

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_experimental.managers.manager_base import _resolve_reset_mask


def test_reset_mask_rejects_wrong_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manager kernels should fail before receiving a mask from another device."""
    owner = SimpleNamespace(num_envs=2, device="expected")
    env_mask = wp.zeros(2, dtype=wp.bool, device="cpu")
    expected_device = SimpleNamespace(alias="expected")
    monkeypatch.setattr(wp, "get_device", lambda device: expected_device)

    with pytest.raises(ValueError, match="env_mask must be on"):
        _resolve_reset_mask(owner, None, env_mask)
