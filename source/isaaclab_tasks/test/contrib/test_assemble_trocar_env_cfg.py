# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the assemble trocar task configuration."""

from isaaclab_newton.sim.schemas import MujocoCollisionCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg

from isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg import G1AssembleTrocarEnvCfg


def test_bore_contact_tuning_preserves_physx_properties():
    """The Newton-only bore override must not replace the existing PhysX contact settings."""
    cfg = G1AssembleTrocarEnvCfg()
    collision_props = cfg.scene.trocar_1.spawn.collision_props

    assert isinstance(collision_props, dict)
    assert any(
        isinstance(fragment, PhysxCollisionCfg) and fragment.contact_offset == 0.001 and fragment.rest_offset == -0.001
        for fragment in collision_props["/.*"]
    )

    bore_patterns = [pattern for pattern in collision_props if pattern != "/.*"]
    assert len(bore_patterns) == 1
    bore_props = collision_props[bore_patterns[0]]
    assert len(bore_props) == 1
    assert isinstance(bore_props[0], MujocoCollisionCfg)
    assert bore_props[0].priority == 1
    assert bore_props[0].solref == (0.002, 1.0)


def test_bore_friction_remap_is_newton_specific():
    """The startup term records the PhysX average-combine equivalent for Newton."""
    cfg = G1AssembleTrocarEnvCfg()
    term = cfg.events.newton_bore_friction

    assert term.mode == "startup"
    assert term.params["asset_cfg"].name == "trocar_1"
    assert term.params["source_mu"] == 1.2
    assert term.params["target_mu"] == 0.65
