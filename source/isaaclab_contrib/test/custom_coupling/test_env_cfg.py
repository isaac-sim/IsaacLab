# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling environment configuration."""

import sys

import pytest

from isaaclab_contrib.custom_coupling.franka_soft_env_cfg import FrankaSoftCustomCouplingEnvCfg, PhysicsCfg

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothCameraEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftCameraEnvCfg, FrankaSoftEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import PhysicsCfg as CorePhysicsCfg
from isaaclab_tasks.utils.hydra import register_task, resolve_presets

MANUAL_MANAGER = "isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
PROXY_MANAGER = "isaaclab_contrib.coupling.coupler:NewtonCouplerManager"


def test_example_default_preset_uses_the_manual_coupler() -> None:
    """Importing the example must select its own manual coupling preset."""
    env_cfg = resolve_presets(FrankaSoftCustomCouplingEnvCfg(), selected=())

    assert env_cfg.sim.physics.class_type == MANUAL_MANAGER
    assert env_cfg.scene.replicate_physics


def test_core_task_default_preset_uses_the_proxy_coupler() -> None:
    """Adding the manual preset in the example must not leak into the core task."""
    env_cfg = resolve_presets(FrankaSoftEnvCfg(), selected=())

    assert env_cfg.sim.physics.class_type == PROXY_MANAGER


def test_core_declares_only_the_proxy_preset() -> None:
    """The manual preset lives in contrib; core declares only the proxy variant."""
    core_variants = set(type(CorePhysicsCfg()).__dataclass_fields__)
    contrib_variants = set(type(PhysicsCfg()).__dataclass_fields__)

    assert "newton_mjwarp_vbd_proxy" in core_variants
    assert "newton_mjwarp_vbd" not in core_variants
    assert "newton_mjwarp_vbd" in contrib_variants


def test_core_task_rejects_the_removed_preset_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting the moved name on a core task must fail loudly, not fall back."""
    monkeypatch.setattr(sys, "argv", ["prog", "presets=newton_mjwarp_vbd"])

    with pytest.raises(ValueError, match="newton_mjwarp_vbd"):
        register_task("Isaac-Lift-Soft-Franka", "rsl_rl_cfg_entry_point")


def test_example_accepts_the_manual_preset_name() -> None:
    """The example declares the name, so the same selection resolves instead of raising."""
    env_cfg = resolve_presets(FrankaSoftCustomCouplingEnvCfg(), selected=("newton_mjwarp_vbd",))

    assert env_cfg.sim.physics.class_type == MANUAL_MANAGER


def test_proxy_preset_selectable_on_example() -> None:
    """The example still resolves the inherited core proxy preset."""
    env_cfg = resolve_presets(FrankaSoftCustomCouplingEnvCfg(), selected=("newton_mjwarp_vbd_proxy",))

    assert env_cfg.sim.physics.class_type == PROXY_MANAGER


@pytest.mark.parametrize("factory", [FrankaSoftCameraEnvCfg, FrankaClothCameraEnvCfg])
@pytest.mark.parametrize("preset", [(), ("newton_mjwarp_vbd_proxy",), ("newton_mjwarp_vbd",)])
def test_camera_tasks_always_use_proxy_coupling(factory, preset: tuple[str, ...]) -> None:
    """Camera tasks only support proxy coupling, including for the contrib preset name."""
    env_cfg = resolve_presets(factory(), selected=preset)

    assert env_cfg.sim.physics.class_type == PROXY_MANAGER
