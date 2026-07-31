# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling environment configuration."""

import warnings

import pytest

from isaaclab_contrib.custom_coupling.franka_soft_env_cfg import FrankaSoftCustomCouplingEnvCfg, PhysicsCfg

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothCameraEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftCameraEnvCfg, FrankaSoftEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import PhysicsCfg as CorePhysicsCfg
from isaaclab_tasks.utils.hydra import resolve_presets

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


def test_legacy_core_preset_name_warns_and_maps_to_proxy() -> None:
    """The removed core name stays usable for one release via the legacy alias."""
    env_cfg = FrankaSoftEnvCfg()

    with pytest.warns(FutureWarning, match="newton_mjwarp_vbd_proxy"):
        resolve_presets(env_cfg, selected=("newton_mjwarp_vbd",))

    assert env_cfg.sim.physics.class_type == PROXY_MANAGER


def test_example_shadows_the_legacy_alias() -> None:
    """The example declares the name itself, so it must not be rewritten to proxy."""
    env_cfg = FrankaSoftCustomCouplingEnvCfg()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        resolve_presets(env_cfg, selected=("newton_mjwarp_vbd",))

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
