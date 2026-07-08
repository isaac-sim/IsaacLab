# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration tests for OVPhysX support in assigned dexterous tasks."""

import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg

from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import (
    ShadowHandEnvCfg,
    ShadowHandOpenAIEnvCfg,
)
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets


@pytest.mark.parametrize(
    "env_cfg_type",
    [
        pytest.param(AllegroHandEnvCfg, id="allegro_direct"),
        pytest.param(ShadowHandEnvCfg, id="shadow_direct"),
        pytest.param(ShadowHandOpenAIEnvCfg, id="shadow_openai_ff"),
        pytest.param(HandoverEnvCfg, id="handover"),
    ],
)
def test_ovphysx_physics_preset_resolves_for_assigned_dexterous_variants(env_cfg_type):
    """Verify every assigned variant resolves its physics preset to OVPhysX."""
    env_cfg = env_cfg_type()

    assert "ovphysx" in collect_presets(env_cfg)["sim.physics"]

    resolved_cfg = resolve_presets(env_cfg, {"ovphysx"})

    assert isinstance(resolved_cfg.sim.physics, OvPhysxCfg)


@pytest.mark.parametrize("env_cfg_type", [HandoverEnvCfg])
def test_newton_physics_and_scene_presets_resolve_for_handover(env_cfg_type):
    """Verify Direct Handover resolves Newton physics and cloning together."""
    resolved_cfg = resolve_presets(env_cfg_type(), {"newton_mjwarp"})

    assert isinstance(resolved_cfg.sim.physics, NewtonCfg)
    assert not resolved_cfg.scene.clone_in_fabric
