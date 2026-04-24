# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the navigation environment configuration.

These tests verify that PresetCfg wrappers inherited from the locomotion
velocity environment are properly resolved in the navigation env config,
and that sensor update periods and physics materials are correctly propagated.
"""

from isaaclab_tasks.utils.hydra import PresetCfg, resolve_presets


class TestNavigationEnvCfg:
    """Regression tests for NavigationEnvCfg preset resolution."""

    def test_low_level_env_cfg_presets_resolved(self):
        """LOW_LEVEL_ENV_CFG should have all PresetCfg wrappers resolved at import time."""
        from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import LOW_LEVEL_ENV_CFG

        assert not isinstance(LOW_LEVEL_ENV_CFG.scene.contact_forces, PresetCfg), (
            "LOW_LEVEL_ENV_CFG.scene.contact_forces should be a resolved ContactSensorCfg, not a PresetCfg wrapper"
        )

    def test_contact_forces_update_period(self):
        """contact_forces.update_period must equal sim.dt after config creation."""
        from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import NavigationEnvCfg

        cfg = NavigationEnvCfg()
        resolved = resolve_presets(cfg)

        assert not isinstance(resolved.scene.contact_forces, PresetCfg)
        assert resolved.scene.contact_forces.update_period == resolved.sim.dt, (
            f"Expected update_period={resolved.sim.dt}, got {resolved.scene.contact_forces.update_period}. "
            "This indicates update_period was set on a PresetCfg wrapper and lost during resolution."
        )

    def test_physics_material_propagated(self):
        """sim.physics_material should match the terrain's physics material."""
        from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import NavigationEnvCfg

        cfg = NavigationEnvCfg()
        resolved = resolve_presets(cfg)

        terrain_mat = resolved.scene.terrain.physics_material
        sim_mat = resolved.sim.physics_material
        assert sim_mat.static_friction == terrain_mat.static_friction, (
            f"sim.physics_material.static_friction ({sim_mat.static_friction}) != "
            f"terrain.physics_material.static_friction ({terrain_mat.static_friction})"
        )
        assert sim_mat.dynamic_friction == terrain_mat.dynamic_friction
        assert sim_mat.restitution == terrain_mat.restitution

    def test_no_remaining_presets(self):
        """No PresetCfg wrappers should remain after resolve_presets."""
        from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import NavigationEnvCfg
        from isaaclab_tasks.utils.hydra import collect_presets

        cfg = NavigationEnvCfg()
        resolved = resolve_presets(cfg)
        remaining = collect_presets(resolved)
        assert len(remaining) == 0, f"Unresolved PresetCfg nodes remain: {list(remaining.keys())}"

    def test_play_config(self):
        """NavigationEnvCfg_PLAY should also resolve correctly."""
        from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import NavigationEnvCfg_PLAY

        cfg = NavigationEnvCfg_PLAY()
        resolved = resolve_presets(cfg)

        assert resolved.scene.num_envs == 50
        assert not resolved.observations.policy.enable_corruption
        assert resolved.scene.contact_forces.update_period == resolved.sim.dt
        assert not isinstance(resolved.scene.contact_forces, PresetCfg)
