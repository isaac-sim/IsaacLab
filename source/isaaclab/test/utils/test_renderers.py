# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for renderer-related utility helpers."""

import pytest

from pxr import Usd, UsdGeom, UsdLux

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


def test_scene_partitions_include_lights_and_remap_native_copies() -> None:
    """Light selectors follow destination worlds while global lights remain shared."""
    from isaaclab.utils.renderers import scene_partition_attributes

    stage = Usd.Stage.CreateInMemory()
    UsdLux.DomeLight.Define(stage, "/World/Shared")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Camera.Define(stage, "/World/envs/env_0/Rig/Camera")
    UsdLux.DomeLight.Define(stage, "/World/envs/env_0/Rig/Dome")
    UsdLux.RectLight.Define(stage, "/World/envs/env_0/Rig/Ceiling")
    UsdGeom.Cube.Define(stage, "/World/envs/env_0/Rig/Object")
    env_paths = [f"/World/envs/env_{i}" for i in range(3)]
    assignments = scene_partition_attributes(
        stage,
        env_paths,
        copies=[("/World/envs/env_0/Rig", ["/World/envs/env_2/Rig", "/World/envs/env_1/Rig"])],
    )
    assert assignments["primvars:omni:scenePartition"] == dict(zip(env_paths, ("env_0", "env_1", "env_2")))
    assert assignments["omni:scenePartition"] == {
        f"/World/envs/env_{i}/Rig/{name}": f"env_{i}" for i in range(3) for name in ("Camera", "Dome", "Ceiling")
    }
