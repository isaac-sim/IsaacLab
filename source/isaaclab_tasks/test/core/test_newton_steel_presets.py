# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolved task settings for Newton Steel."""

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, SteelSolverCfg

from isaaclab.sim import SimulationCfg

from isaaclab_tasks.contrib.keyboard.so101_env_cfg import SO101KeyboardEnvCfg, SO101SceneCfg
from isaaclab_tasks.utils import PresetCfg, resolve_task_config


def test_so101_newton_steel_preset_enables_parallel_streams() -> None:
    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101", None, overrides=["presets=newton_steel"])
    assert cfg.scene.num_envs == 4096
    assert isinstance(cfg.sim.physics.solver_cfg, SteelSolverCfg)
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.sim.physics.solver_cfg.use_parallel_streams is True
    assert cfg.sim.physics.solver_cfg.parallel_tree is True
    assert cfg.sim.physics.collision_cfg.rigid_contact_max == 170 * 4096
    assert cfg.sim.physics.default_shape_cfg.gap == 0.002


@pytest.mark.parametrize(
    ("override", "contact_max"),
    [
        ("env.sim.physics.collision_cfg=null", None),
        ("env.sim.physics.collision_cfg.rigid_contact_max=1000000", 1000000),
    ],
)
def test_so101_steel_contact_budget_respects_cli_override(override, contact_max):
    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101", None, overrides=["presets=newton_steel", override])
    if contact_max is None:
        assert cfg.sim.physics.collision_cfg is None
    else:
        assert cfg.sim.physics.collision_cfg.rigid_contact_max == contact_max


def test_so101_steel_contact_budget_scales_with_constructed_scene():
    cfg = SO101KeyboardEnvCfg(scene=SO101SceneCfg(num_envs=496, env_spacing=1.0))
    assert cfg.sim.physics.newton_steel.collision_cfg.rigid_contact_max == 170 * 496


@pytest.mark.parametrize(
    ("task", "iterations", "solver_changes"),
    [
        (
            "Isaac-Reach-Franka",
            4,
            {
                "update_mass_matrix_interval": 4,
                "articulation_constraint_max": 16,
                "enable_joint_velocity_limits": False,
            },
        ),
        (
            "Isaac-Reach-Franka-OSC",
            8,
            {
                "update_mass_matrix_interval": 2,
                "articulation_constraint_max": 64,
                "enable_joint_velocity_limits": True,
            },
        ),
        (
            "Isaac-Reach-UR10",
            4,
            {"filter_fixed_root_contacts": True, "update_mass_matrix_interval": 4, "articulation_constraint_max": 16},
        ),
        ("Isaac-Reorient-Cube-Allegro", 8, {"contact_friction_gap_max": 0.001, "contact_torsion": False}),
        ("Isaac-Reorient-Cube-Allegro-Direct", 8, {"contact_friction_gap_max": 0.001, "contact_torsion": False}),
        ("Isaac-Velocity-Rough-AnymalD", 16, {}),
        ("Isaac-Velocity-Flat-AnymalD", 8, {}),
        ("IsaacContrib-Velocity-Flat-AnymalC", 16, {}),
        ("Isaac-Velocity-Rough-UnitreeGo2", 8, {"joint_limit_activation_gap": 0.4}),
        ("Isaac-Velocity-Flat-UnitreeGo2", 8, {"joint_limit_activation_gap": 0.2}),
    ],
)
def test_newton_steel_tuning_preserves_substeps(task, iterations, solver_changes):
    cfg, _ = resolve_task_config(task, None, overrides=["presets=newton_steel"])
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.sim.physics.solver_cfg.iterations == iterations
    for field, expected in solver_changes.items():
        assert getattr(cfg.sim.physics.solver_cfg, field) == expected


@pytest.mark.parametrize("task", ["Isaac-Reach-Franka", "Isaac-Open-Drawer-Franka", "Isaac-Open-Drawer-Franka-Direct"])
def test_franka_newton_steel_tasks_select_driven_usd_variant(task: str) -> None:
    cfg, _ = resolve_task_config(task, None, overrides=["presets=newton_steel"])
    assert cfg.scene.robot.spawn.variants["Physics"] == "physx"


@pytest.mark.parametrize("task", ["Isaac-Open-Drawer-Franka", "Isaac-Open-Drawer-Franka-Direct"])
@pytest.mark.parametrize(
    ("override", "dt"),
    [("presets=newton_mjwarp", 1 / 600), ("presets=newton_steel", 1 / 60), ("physics=newton_steel", 1 / 60)],
)
def test_cabinet_simulation_resolves_from_physics_and_dt_presets(task: str, override: str, dt: float) -> None:
    cfg, _ = resolve_task_config(task, None, overrides=[override])
    assert isinstance(cfg.sim, SimulationCfg)
    assert not isinstance(cfg.sim, PresetCfg)
    assert cfg.sim.dt == dt
    expected_solver = MJWarpSolverCfg if override == "presets=newton_mjwarp" else SteelSolverCfg
    assert isinstance(cfg.sim.physics.solver_cfg, expected_solver)
    assert cfg.decimation == (10 if override == "presets=newton_mjwarp" else 1)
