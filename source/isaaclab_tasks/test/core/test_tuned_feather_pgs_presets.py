# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture and configuration gates for tuned FeatherPGS task presets."""

import math

import pytest
from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonCfg

from isaaclab_tasks.contrib.velocity.config.a1 import rough_env_cfg as a1_rough_env_cfg
from isaaclab_tasks.contrib.velocity.config.a1.flat_env_cfg import PhysicsCfg as A1FlatPhysicsCfg
from isaaclab_tasks.contrib.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
from isaaclab_tasks.contrib.velocity.config.a1.rough_env_cfg import UnitreeA1RoughEnvCfg
from isaaclab_tasks.contrib.velocity.config.anymal_b.flat_env_cfg import AnymalBFlatEnvCfg
from isaaclab_tasks.contrib.velocity.config.anymal_b.flat_env_cfg import PhysicsCfg as AnymalBFlatPhysicsCfg
from isaaclab_tasks.contrib.velocity.config.anymal_b.rough_env_cfg import AnymalBRoughEnvCfg
from isaaclab_tasks.contrib.velocity.config.go1 import rough_env_cfg as go1_rough_env_cfg
from isaaclab_tasks.contrib.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg
from isaaclab_tasks.core.cabinet.cabinet_env_cfg import CabinetEnvCfg, CabinetSimCfg
from isaaclab_tasks.core.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
    KukaAllegroMixinCfg,
)
from isaaclab_tasks.core.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg
from isaaclab_tasks.core.reach.config.ur_10.joint_pos_env_cfg import UR10ReachEnvCfg
from isaaclab_tasks.core.reach.reach_env_cfg import ReachPhysicsCfg
from isaaclab_tasks.core.velocity.config.cassie.flat_env_cfg import CassieFlatEnvCfg
from isaaclab_tasks.core.velocity.config.cassie.flat_env_cfg import PhysicsCfg as CassieFlatPhysicsCfg
from isaaclab_tasks.core.velocity.config.h1 import rough_env_cfg as h1_rough_env_cfg
from isaaclab_tasks.core.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.h1.flat_env_cfg import PhysicsCfg as H1FlatPhysicsCfg
from isaaclab_tasks.core.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


@pytest.mark.parametrize(
    (
        "env_cfg_type",
        "mass_matrix_interval",
        "joint_limit_gap",
        "dense_capacity",
        "matrix_free_capacity",
        "hinv_jt_kernel",
        "use_cuda_graph",
        "terrain_contacts",
    ),
    (
        (DexsuiteKukaAllegroLiftEnvCfg, 2, 0.1, 384, 64, "par_row", True, False),
        (UnitreeA1FlatEnvCfg, 1, 0.1, 64, 512, "auto", False, True),
        (AnymalBFlatEnvCfg, 1, math.inf, 128, 32, "auto", True, True),
        (AnymalBRoughEnvCfg, 2, 0.1, 128, 32, "auto", True, True),
        (CassieFlatEnvCfg, 1, math.inf, 64, 512, "auto", True, True),
        (FrankaReachEnvCfg, 1, 0.1, 64, 32, "auto", False, False),
        (UR10ReachEnvCfg, 1, 0.1, 32, 32, "auto", True, False),
        (CabinetEnvCfg, 1, math.inf, 256, 32, "par_row", True, False),
    ),
)
def test_tuned_feather_pgs_recipe_is_owned_by_each_task(
    env_cfg_type,
    mass_matrix_interval,
    joint_limit_gap,
    dense_capacity,
    matrix_free_capacity,
    hinv_jt_kernel,
    use_cuda_graph,
    terrain_contacts,
):
    """Each tuned task should resolve its measured capacity-safe solver recipe."""
    cfg = resolve_presets(env_cfg_type(), selected=("feather_pgs",))
    physics = cfg.sim.physics

    assert isinstance(physics, NewtonCfg)
    assert physics.num_substeps == 1
    assert physics.use_cuda_graph is use_cuda_graph
    solver = physics.solver_cfg
    assert isinstance(solver, FeatherPGSSolverCfg)
    assert solver.pgs_mode == "matrix_free"
    assert solver.update_mass_matrix_interval == mass_matrix_interval
    assert solver.enable_joint_limits is True
    assert solver.joint_limit_activation_gap == joint_limit_gap
    assert solver.pgs_iterations == 8
    assert solver.pgs_velocity_iterations == 0
    assert solver.dense_max_constraints == dense_capacity
    assert solver.mf_max_constraints == matrix_free_capacity
    assert solver.hinv_jt_kernel == hinv_jt_kernel
    assert solver.pgs_warmstart is False
    assert solver.pgs_omega == 1.0
    assert solver.pgs_beta == 0.05
    assert solver.pgs_cfm == 1.0e-6
    assert solver.serial_kernel_block_dim == 64
    assert solver.row_watermark is False
    if terrain_contacts:
        assert physics.collision_cfg.max_triangle_pairs == 2_500_000
        assert physics.default_shape_cfg.margin == 0.01


@pytest.mark.parametrize(
    ("env_cfg_type", "num_substeps", "pgs_beta", "pgs_cfm", "pgs_omega"),
    (
        (UnitreeA1RoughEnvCfg, 1, 0.02, 1.0e-5, 0.8),
        (UnitreeGo1RoughEnvCfg, 1, 0.02, 1.0e-5, 0.8),
        (H1RoughEnvCfg, 1, 0.05, 1.0e-6, 1.0),
        (H1FlatEnvCfg, 1, 0.05, 1.0e-6, 1.0),
    ),
)
def test_validated_locomotion_presets_match_matrix_free_recipe(
    env_cfg_type, num_substeps, pgs_beta, pgs_cfm, pgs_omega
):
    """The four validated locomotion tasks should retain their measured matrix-free recipe."""
    cfg = resolve_presets(env_cfg_type(), selected=("feather_pgs",))
    physics = cfg.sim.physics
    solver = physics.solver_cfg

    assert isinstance(physics, NewtonCfg)
    assert physics.num_substeps == num_substeps
    assert physics.use_cuda_graph is True
    assert isinstance(solver, FeatherPGSSolverCfg)
    assert solver.pgs_mode == "matrix_free"
    assert solver.update_mass_matrix_interval == 1
    assert solver.enable_joint_limits is True
    assert solver.pgs_iterations == 8
    assert solver.pgs_velocity_iterations == 0
    assert solver.dense_max_constraints == 64
    assert solver.mf_max_constraints == 512
    assert solver.pgs_warmstart is False
    assert solver.pgs_omega == pytest.approx(pgs_omega)
    assert solver.pgs_beta == pytest.approx(pgs_beta)
    assert solver.pgs_cfm == pytest.approx(pgs_cfm)
    assert solver.serial_kernel_block_dim == 64


@pytest.mark.parametrize(
    ("env_cfg_type", "angular_damping", "velocity_limits", "velocity_limit_fraction", "reduce_contacts"),
    (
        (UnitreeA1RoughEnvCfg, 0.6, True, 0.35, False),
        (UnitreeGo1RoughEnvCfg, 0.6, True, 0.35, False),
        (H1RoughEnvCfg, 0.05, False, 0.0, True),
    ),
)
def test_validated_rough_locomotion_presets_match_stabilization_recipe(
    env_cfg_type, angular_damping, velocity_limits, velocity_limit_fraction, reduce_contacts
):
    """Rough-terrain task presets should retain their validated stabilization and capacities."""
    cfg = resolve_presets(env_cfg_type(), selected=("feather_pgs",))
    physics = cfg.sim.physics
    solver = physics.solver_cfg

    assert solver.angular_damping == pytest.approx(angular_damping)
    assert solver.enable_joint_velocity_limits is velocity_limits
    assert solver.velocity_limit_activation_fraction == pytest.approx(velocity_limit_fraction)
    assert solver.pgs_velocity_drive_mode == "freeze"
    assert physics.collision_cfg.reduce_contacts is reduce_contacts
    assert physics.collision_cfg.rigid_contact_max is None
    assert physics.collision_cfg.max_triangle_pairs == 2_500_000
    assert physics.default_shape_cfg.margin == 0.01


def test_h1_flat_preserves_validated_stabilization_recipe():
    """H1 flat should retain the measured stabilization used by its single-substep recipe."""
    cfg = resolve_presets(H1FlatEnvCfg(), selected=("feather_pgs",))
    solver = cfg.sim.physics.solver_cfg

    assert solver.angular_damping == pytest.approx(0.05)
    assert solver.pgs_velocity_drive_mode == "freeze"
    assert solver.joint_limit_activation_gap == pytest.approx(0.1)


def test_a1_rough_uses_default_mesh_bvh_constructor():
    """A1 rough should retain the stable default mesh BVH selection."""
    cfg = resolve_presets(UnitreeA1RoughEnvCfg(), selected=("feather_pgs",))

    assert cfg.sim.physics.mesh_bvh_constructor is None


def test_tuned_feather_pgs_presets_have_task_local_ownership():
    """Task-specific capacities must not leak into a generic composition root."""
    assert "feather_pgs" in A1FlatPhysicsCfg.__annotations__
    assert "feather_pgs" in AnymalBFlatPhysicsCfg.__annotations__
    assert "feather_pgs" in CassieFlatPhysicsCfg.__annotations__
    assert AnymalBRoughEnvCfg.PhysicsCfg.__qualname__.endswith("AnymalBRoughEnvCfg.PhysicsCfg")
    assert "feather_pgs" in AnymalBRoughEnvCfg.PhysicsCfg.__annotations__
    assert UnitreeA1RoughEnvCfg.PhysicsCfg.__qualname__.endswith("UnitreeA1RoughEnvCfg.PhysicsCfg")
    assert "feather_pgs" in UnitreeA1RoughEnvCfg.PhysicsCfg.__annotations__
    assert UnitreeGo1RoughEnvCfg.PhysicsCfg.__qualname__.endswith("UnitreeGo1RoughEnvCfg.PhysicsCfg")
    assert "feather_pgs" in UnitreeGo1RoughEnvCfg.PhysicsCfg.__annotations__
    assert H1RoughEnvCfg.PhysicsCfg.__qualname__.endswith("H1RoughEnvCfg.PhysicsCfg")
    assert "feather_pgs" in H1RoughEnvCfg.PhysicsCfg.__annotations__
    assert "feather_pgs" in H1FlatPhysicsCfg.__annotations__
    assert not hasattr(a1_rough_env_cfg, "PhysicsCfg")
    assert not hasattr(go1_rough_env_cfg, "PhysicsCfg")
    assert not hasattr(h1_rough_env_cfg, "PhysicsCfg")
    assert FrankaReachEnvCfg.PhysicsCfg.__qualname__.endswith("FrankaReachEnvCfg.PhysicsCfg")
    assert "feather_pgs" in FrankaReachEnvCfg.PhysicsCfg.__annotations__
    assert UR10ReachEnvCfg.PhysicsCfg.__qualname__.endswith("UR10ReachEnvCfg.PhysicsCfg")
    assert "feather_pgs" in UR10ReachEnvCfg.PhysicsCfg.__annotations__
    assert KukaAllegroMixinCfg.PhysicsCfg.__qualname__.endswith("KukaAllegroMixinCfg.PhysicsCfg")
    assert "feather_pgs" in KukaAllegroMixinCfg.PhysicsCfg.__annotations__
    assert "feather_pgs" in CabinetSimCfg.__annotations__
    assert "feather_pgs" not in ReachPhysicsCfg.__annotations__


@pytest.mark.parametrize(
    ("env_cfg_type", "actuator_name", "expected_armature"),
    (
        (UnitreeA1FlatEnvCfg, "base_legs", 0.05),
        (UnitreeA1RoughEnvCfg, "base_legs", 0.05),
        (UnitreeGo1RoughEnvCfg, "base_legs", 0.10),
        (AnymalBFlatEnvCfg, "legs", 0.30),
        (AnymalBRoughEnvCfg, "legs", 0.30),
        (CassieFlatEnvCfg, "legs", 0.05),
    ),
)
def test_tuned_locomotion_presets_preserve_validated_armature(env_cfg_type, actuator_name, expected_armature):
    """The solver tuning should retain the armatures used by the validated runs."""
    cfg = resolve_presets(env_cfg_type(), selected=("feather_pgs",))

    assert cfg.scene.robot.actuators[actuator_name].armature == expected_armature
