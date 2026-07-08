# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration tests for OVPhysX support in assigned dexterous tasks."""

import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg

from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg
from isaaclab_tasks.core.handover.handover_manager_env_cfg import HandoverManagerEnvCfg
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_manager_env_cfg import (
    AllegroCubeEnvCfg,
    AllegroCubeEnvCfg_PLAY,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env_cfg import (
    ShadowHandCameraBenchmarkEnvCfg,
    ShadowHandCameraEnvCfg,
    ShadowHandCameraEnvPlayCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_manager_env_cfg import (
    ShadowHandCameraManagerBenchmarkEnvCfg,
    ShadowHandCameraManagerEnvCfg,
    ShadowHandCameraManagerPlayEnvCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import (
    ShadowHandEnvCfg,
    ShadowHandOpenAIEnvCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    ShadowHandManagerEnvCfg,
    ShadowHandOpenAIManagerEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets


@pytest.mark.parametrize(
    "env_cfg_type",
    [
        pytest.param(AllegroHandEnvCfg, id="allegro_direct"),
        pytest.param(ShadowHandEnvCfg, id="shadow_direct"),
        pytest.param(ShadowHandOpenAIEnvCfg, id="shadow_openai_ff"),
        pytest.param(ShadowHandCameraEnvCfg, id="shadow_camera"),
        pytest.param(ShadowHandCameraEnvPlayCfg, id="shadow_camera_play"),
        pytest.param(ShadowHandCameraBenchmarkEnvCfg, id="shadow_camera_benchmark"),
        pytest.param(ShadowHandCameraManagerEnvCfg, id="shadow_camera_manager"),
        pytest.param(ShadowHandCameraManagerPlayEnvCfg, id="shadow_camera_manager_play"),
        pytest.param(ShadowHandCameraManagerBenchmarkEnvCfg, id="shadow_camera_manager_benchmark"),
        pytest.param(HandoverEnvCfg, id="handover"),
        pytest.param(HandoverManagerEnvCfg, id="handover_manager"),
        pytest.param(AllegroCubeEnvCfg, id="allegro_manager"),
        pytest.param(AllegroCubeEnvCfg_PLAY, id="allegro_manager_play"),
        pytest.param(ShadowHandManagerEnvCfg, id="shadow_manager"),
        pytest.param(ShadowHandOpenAIManagerEnvCfg, id="shadow_openai_manager"),
    ],
)
def test_ovphysx_physics_preset_resolves_for_assigned_dexterous_variants(env_cfg_type):
    """Verify every assigned variant resolves its physics preset to OVPhysX."""
    env_cfg = env_cfg_type()

    assert "ovphysx" in collect_presets(env_cfg)["sim.physics"]

    resolved_cfg = resolve_presets(env_cfg, {"ovphysx"})

    assert isinstance(resolved_cfg.sim.physics, OvPhysxCfg)


@pytest.mark.parametrize("env_cfg_type", [AllegroCubeEnvCfg, AllegroCubeEnvCfg_PLAY])
def test_newton_physics_and_scene_presets_resolve_for_allegro_manager(env_cfg_type):
    """Verify Allegro Manager uses Newton physics, an articulated cube, and non-Fabric cloning."""
    resolved_cfg = resolve_presets(env_cfg_type(), {"newton_mjwarp"})

    assert isinstance(resolved_cfg.sim.physics, NewtonCfg)
    assert isinstance(resolved_cfg.scene.object, ArticulationCfg)
    assert not resolved_cfg.scene.clone_in_fabric


@pytest.mark.parametrize("env_cfg_type", [HandoverEnvCfg, HandoverManagerEnvCfg])
def test_newton_physics_and_scene_presets_resolve_for_handover(env_cfg_type):
    """Verify Direct and Manager Handover resolve Newton physics and cloning together."""
    resolved_cfg = resolve_presets(env_cfg_type(), {"newton_mjwarp"})

    assert isinstance(resolved_cfg.sim.physics, NewtonCfg)
    assert not resolved_cfg.scene.clone_in_fabric


def test_default_physics_scene_preset_resolves_play_env_count_for_allegro_manager():
    """Verify the default Allegro Manager play scene retains its reduced environment count."""
    resolved_cfg = resolve_presets(AllegroCubeEnvCfg_PLAY(), set())

    assert resolved_cfg.scene.num_envs == 50
