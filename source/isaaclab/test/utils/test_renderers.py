# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for renderer-related utility helpers."""

import pytest

from isaaclab.utils.renderers import isaac_rtx_per_env_scene_partition_enabled

_ENV_VAR = "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION"


def test_scene_partitioning_is_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scene partitioning should be enabled when the environment variable is absent."""
    monkeypatch.delenv(_ENV_VAR, raising=False)

    assert isaac_rtx_per_env_scene_partition_enabled()


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_scene_partitioning_accepts_binary_override(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """Explicit binary environment values should control scene partitioning."""
    monkeypatch.setenv(_ENV_VAR, value)

    assert isaac_rtx_per_env_scene_partition_enabled() is expected


@pytest.mark.parametrize("value", ["", "true", "yes", "on", "2"])
def test_scene_partitioning_rejects_invalid_override(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Invalid environment values should fail instead of silently disabling partitioning."""
    monkeypatch.setenv(_ENV_VAR, value)

    with pytest.raises(ValueError, match=f"Invalid value for {_ENV_VAR}"):
        isaac_rtx_per_env_scene_partition_enabled()
