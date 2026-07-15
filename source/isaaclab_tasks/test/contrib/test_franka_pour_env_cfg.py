# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction tests for the Franka pour env config (no simulator)."""

import ast
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from isaaclab_contrib.coupling import CouplerProxyCfg, NewtonCoupler

import isaaclab_tasks.contrib.franka_pour as franka_pour
import isaaclab_tasks.contrib.franka_pour.config.franka  # noqa: F401
import isaaclab_tasks.contrib.franka_pour.mdp as mdp
import isaaclab_tasks.contrib.franka_pour.pour_env_cfg as pour_env_cfg
from isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaPourPPORunnerCfg,
    FrankaPourResetDatasetPPORunnerCfg,
    FrankaPourResetMixturePPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_pour.cube_bowl_spawner_cfg import CubeBowlSpawnerCfg
from isaaclab_tasks.contrib.franka_pour.cup_media import cup_cavity_lattice
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
    FrankaPourEnvCfg,
    FrankaPourEnvCfg_PLAY,
    FrankaPourEnvCfg_RESET_DATASET,
    FrankaPourEnvCfg_RESET_DATASET_EVAL,
    FrankaPourEnvCfg_RESET_DATASET_PLAY,
    FrankaPourEnvCfg_RESET_MIXTURE,
    FrankaPourEnvCfg_RESET_MIXTURE_EVAL,
    FrankaPourEnvCfg_RESET_MIXTURE_PLAY,
    FrankaPourEnvCfg_TELEOP,
    _resolve_mpm_cell_cap,
)
from isaaclab_tasks.contrib.franka_pour.reset_utils import (
    balanced_cyclic_permutations,
    boolean_selection_mask,
    randomization_extent_index_pools,
    sample_index_pools,
    target_xy_behind_source,
)


def test_boolean_selection_mask_preserves_device_shape_and_dtype():
    selected = torch.tensor([[1, 3], [3, 4]], dtype=torch.long)

    mask = boolean_selection_mask(6, selected)

    assert mask.device == selected.device
    assert mask.dtype == torch.bool
    assert mask.shape == (6,)
    assert mask.tolist() == [False, True, False, True, True, False]


def test_franka_pour_config_import_does_not_preload_usd_before_kit():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import isaaclab_tasks.contrib.franka_pour.pour_env_cfg; "
                "assert 'newton' not in sys.modules; assert 'pxr' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_balanced_cyclic_permutations_preserve_marginals_and_balance_pairings():
    values = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0])

    permutations = balanced_cyclic_permutations(values, group_count=49)

    assert permutations.shape == (49, 5)
    expected = torch.sort(values).values.expand(49, -1)
    torch.testing.assert_close(torch.sort(permutations, dim=-1).values, expected)
    for column in range(permutations.shape[1]):
        _, counts = torch.unique(permutations[:, column], return_counts=True)
        assert int(counts.amax() - counts.amin()) <= 1
    assert torch.unique(permutations, dim=0).shape[0] == values.numel()


def test_randomization_extent_pools_combine_all_axes_and_are_nested():
    source_positions = torch.tensor([[0.50, 0.00], [0.525, 0.00], [0.55, 0.00], [0.575, 0.00], [0.60, 0.00]])
    target_positions = torch.tensor([[0.50, -0.20], [0.51, -0.20], [0.54, -0.20], [0.52, -0.20], [0.50, -0.15]])
    source_yaws = torch.tensor([0.00, 0.10, 0.05, 0.15, 0.02])
    tcp_jitter = torch.tensor(
        [[0.00, 0.00, 0.00], [0.012, 0.00, 0.00], [0.006, 0.00, 0.00], [0.018, 0.00, 0.00], [0.004, 0.00, 0.00]]
    )

    pools = randomization_extent_index_pools(
        source_positions,
        source_yaws,
        target_positions,
        tcp_jitter,
        source_center=(0.50, 0.00),
        source_half_range=(0.10, 0.10),
        source_yaw_half_range=0.20,
        target_center=(0.50, -0.20),
        target_half_range=(0.05, 0.05),
        tcp_jitter_half_range=(0.02, 0.02, 0.02),
        extent_levels=(0.5, 0.8, 1.0),
    )

    assert [pool.tolist() for pool in pools] == [[0], [0, 1, 2], [0, 1, 2, 3, 4]]
    assert bool(torch.all(torch.isin(pools[0], pools[1])))
    assert bool(torch.all(torch.isin(pools[1], pools[2])))
    torch.testing.assert_close(pools[2], torch.arange(5))


def test_randomization_extent_pools_handle_zero_range_axes_without_nan():
    source_positions = torch.tensor([[0.5, -0.1], [0.5, 0.0], [0.5, 0.1], [0.5, 0.0]])
    target_positions = torch.tensor([[0.4, -0.2], [0.4, -0.1], [0.4, 0.0], [0.4, -0.1]])
    source_yaws = torch.zeros(4)
    tcp_jitter = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])

    pools = randomization_extent_index_pools(
        source_positions,
        source_yaws,
        target_positions,
        tcp_jitter,
        source_center=(0.5, 0.0),
        source_half_range=(0.0, 0.1),
        source_yaw_half_range=0.0,
        target_center=(0.4, -0.1),
        target_half_range=(0.0, 0.1),
        tcp_jitter_half_range=(0.0, 0.0, 0.0),
        extent_levels=(0.5, 1.0),
    )

    assert [pool.tolist() for pool in pools] == [[1], [0, 1, 2]]


def test_randomization_extent_pools_include_source_yaw():
    pools = randomization_extent_index_pools(
        torch.zeros((3, 2)),
        torch.tensor([0.0, 0.05, -0.10]),
        torch.zeros((3, 2)),
        torch.zeros((3, 3)),
        source_center=(0.0, 0.0),
        source_half_range=(0.0, 0.0),
        source_yaw_half_range=0.10,
        target_center=(0.0, 0.0),
        target_half_range=(0.0, 0.0),
        tcp_jitter_half_range=(0.0, 0.0, 0.0),
        extent_levels=(0.5, 1.0),
    )

    assert [pool.tolist() for pool in pools] == [[0, 1], [0, 1, 2]]


def test_randomization_extent_pools_require_aligned_bank_rows():
    with pytest.raises(ValueError, match="same row count"):
        randomization_extent_index_pools(
            torch.zeros((2, 3)),
            torch.zeros(2),
            torch.zeros((1, 3)),
            torch.zeros((2, 3)),
            source_center=(0.0, 0.0),
            source_half_range=(0.1, 0.1),
            source_yaw_half_range=0.1,
            target_center=(0.0, 0.0),
            target_half_range=(0.1, 0.1),
            tcp_jitter_half_range=(0.1, 0.1, 0.1),
            extent_levels=(1.0,),
        )


def test_index_pool_sampling_maps_local_slots_back_to_global_bank_rows(monkeypatch):
    pools = (torch.tensor([4, 8]), torch.tensor([1, 5, 9]), torch.tensor([0, 3, 6, 9]))
    pool_ids = torch.tensor([0, 2, 1, 0, 2])

    def sample_last_slot(high, size, *, device):
        return torch.full(size, high - 1, device=device, dtype=torch.long)

    monkeypatch.setattr(torch, "randint", sample_last_slot)
    sampled = sample_index_pools(pools, pool_ids)

    assert sampled.tolist() == [8, 9, 9, 8, 9]


def test_target_randomization_is_bounded_and_separated_behind_source():
    azimuth = math.radians(20.0)
    source_xy = torch.tensor(
        [
            [0.50 * math.cos(azimuth), -0.50 * math.sin(azimuth)],
            [0.68 * math.cos(azimuth), 0.68 * math.sin(azimuth)],
        ]
    )
    unit_samples = torch.tensor([[0.0, 1.0], [1.0, 1.0]])

    minimum_y_separation = torch.tensor([0.129, 0.150])
    target_xy = target_xy_behind_source(
        source_xy,
        target_center=(0.50, -0.18),
        target_half_range=(0.08, 0.30),
        minimum_y_separation=minimum_y_separation,
        unit_samples=unit_samples,
    )

    assert bool(torch.all(target_xy[:, 0] >= 0.42))
    assert bool(torch.all(target_xy[:, 0] <= 0.58))
    assert bool(torch.all(target_xy[:, 1] >= -0.48))
    assert bool(torch.all(target_xy[:, 1] <= 0.12))
    assert bool(torch.all(source_xy[:, 1] - target_xy[:, 1] >= minimum_y_separation - 1.0e-6))


def test_target_randomization_rejects_a_source_position_without_feasible_separation():
    source_xy = torch.tensor([[0.50, -0.40]])

    with pytest.raises(ValueError, match="No target y-position"):
        target_xy_behind_source(
            source_xy,
            target_center=(0.50, -0.18),
            target_half_range=(0.08, 0.30),
            minimum_y_separation=0.129,
            unit_samples=torch.tensor([[0.5, 0.5]]),
        )


def test_finalize_builds_scene_assets_without_mutating_the_caller():
    original = FrankaPourEnvCfg()
    original.scene.num_envs = 8

    resolved = original.finalize()

    assert original.scene.source_cup is None
    assert original.scene.target_cup is None
    assert original.scene.media is None
    assert isinstance(resolved.scene.source_cup, RigidObjectCfg)
    assert isinstance(resolved.scene.target_cup, RigidObjectCfg)
    assert isinstance(resolved.scene.media, MPMObjectCfg)
    assert resolved is not original
    assert resolved.scene is not original.scene
    assert resolved.sim.physics.solver_cfg.scene_cfg is resolved.scene
    assert _media_capacity(resolved) == 8 * 512


def test_finalize_preserves_environment_spacing_for_large_sparse_batches():
    cfg = FrankaPourEnvCfg()
    original_spacing = cfg.scene.env_spacing
    cfg.scene.num_envs = 3068

    resolved = cfg.finalize()

    assert cfg.scene.env_spacing == pytest.approx(original_spacing)
    assert resolved.scene.env_spacing == pytest.approx(original_spacing)
    assert FrankaPourEnvCfg_PLAY().finalize().scene.env_spacing == pytest.approx(original_spacing)


def test_finalize_propagates_final_cup_overrides_to_fresh_assets():
    original = FrankaPourEnvCfg()
    original.source_cup_inner_width = 0.041
    original.target_cup_inner_depth = 0.153
    original.cup_grasp_box_half = (0.0275, 0.028, 0.0595)
    original.gripper_preload_pos = 0.017
    original.cup_grasp_box_friction = 2.4
    original.target_cup_friction = 1.3
    original.cup_mass = 0.071
    original.cup_reset_pos = (0.43, 0.02, 0.01)
    original.target_cup_reset_pos = (0.47, -0.21, 0.02)
    original.arm_home = (0.1, 0.2, 0.3, -2.4, 0.5, 3.2, 0.7)
    original.success_dwell_time_s = 0.10
    # This test intentionally moves the nominal cup independently of the task's base-centred
    # polar workspace. Use the supported rectangular fallback so an unrelated asset override
    # does not need to redefine the polar sector as well.
    original.curriculum_randomized_source_radius_range = None
    original.curriculum_randomized_source_position_range = (0.0, 0.0)
    original.curriculum_randomized_carry_position_range = (0.0, 0.0)

    first = original.finalize()
    first_source = first.scene.source_cup
    first_target = first.scene.target_cup
    assert isinstance(first_source.spawn, CubeBowlSpawnerCfg)
    assert isinstance(first_target.spawn, CubeBowlSpawnerCfg)
    assert first_source.spawn.inner_width == pytest.approx(0.041)
    assert first_target.spawn.inner_depth == pytest.approx(0.153)
    assert first_source.spawn.grasp_proxy_half_extents == pytest.approx((0.0275, 0.028, 0.0595))
    assert first_source.spawn.physics_material.static_friction == pytest.approx(2.4)
    assert first_source.spawn.physics_material.dynamic_friction == pytest.approx(2.4)
    assert first_target.spawn.physics_material.static_friction == pytest.approx(1.3)
    assert first_target.spawn.physics_material.dynamic_friction == pytest.approx(1.3)
    assert first_source.spawn.mass_props.mass == pytest.approx(0.071)
    assert first_source.init_state.pos == pytest.approx((0.43, 0.02, 0.01))
    assert first_source.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert first_target.init_state.pos == pytest.approx((0.47, -0.21, 0.02))
    assert first_target.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert [first.scene.robot.init_state.joint_pos[f"panda_joint{i}"] for i in range(1, 8)] == pytest.approx(
        original.arm_home
    )
    assert first.terminations.success.params["dwell_time_s"] == pytest.approx(0.10)
    assert first.actions.gripper_action.close_position == pytest.approx(0.014)
    assert first.rewards.task_progress.params["grasp_preload_position"] == pytest.approx(0.017)
    expected_contact_command = 0.028 - first.actions.gripper_action.contact_min_deflection
    assert first.rewards.task_progress.params["max_gripper_command"] == pytest.approx(expected_contact_command)
    assert first.rewards.delivered.params["max_gripper_command"] == pytest.approx(expected_contact_command)
    assert first.terminations.success.params["max_gripper_command"] == pytest.approx(expected_contact_command)

    first.source_cup_inner_width = 0.049
    first.cup_grasp_box_half = (0.0315, 0.028, 0.0595)
    second = first.finalize()

    assert second.scene.source_cup is not first.scene.source_cup
    assert second.scene.target_cup is not first.scene.target_cup
    assert second.scene.media is not first.scene.media
    assert first.scene.source_cup.spawn.inner_width == pytest.approx(0.041)
    assert second.scene.source_cup.spawn.inner_width == pytest.approx(0.049)


def test_resolved_cups_have_backend_neutral_rigid_properties():
    resolved = FrankaPourEnvCfg().finalize()
    source = resolved.scene.source_cup
    target = resolved.scene.target_cup

    assert source.prim_path == "{ENV_REGEX_NS}/SourceCup"
    assert isinstance(source.spawn.rigid_props, UsdPhysicsRigidBodyCfg)
    assert source.spawn.rigid_props.rigid_body_enabled is True
    assert source.spawn.rigid_props.kinematic_enabled is False
    assert isinstance(source.spawn.collision_props, UsdPhysicsCollisionCfg)
    assert source.spawn.collision_props.collision_enabled is True
    assert isinstance(source.spawn.mass_props, MassCfg)
    assert isinstance(source.spawn.physics_material, RigidBodyMaterialBaseCfg)
    assert source.spawn.grasp_proxy_half_extents == resolved.cup_grasp_box_half

    assert target.prim_path == "{ENV_REGEX_NS}/TargetCup"
    assert isinstance(target.spawn.rigid_props, UsdPhysicsRigidBodyCfg)
    assert target.spawn.rigid_props.rigid_body_enabled is True
    assert target.spawn.rigid_props.kinematic_enabled is True
    assert target.spawn.grasp_proxy_half_extents is None
    assert isinstance(target.spawn.physics_material, RigidBodyMaterialBaseCfg)


def test_robot_authors_mujoco_gravity_compensation_with_joint_fragment():
    fragments = FrankaPourEnvCfg().scene.robot.spawn.joint_drive_props

    assert isinstance(fragments, list)
    assert any(isinstance(fragment, MujocoJointCfg) and fragment.actuatorgravcomp is True for fragment in fragments)


def test_robot_uses_authored_pour_asset_actuator_defaults_without_mutating_global_franka_preset():
    robot = FrankaPourEnvCfg().finalize().scene.robot

    assert robot.spawn.usd_path == pour_env_cfg.FRANKA_POUR_ROBOT_USD_PATH
    assert robot.spawn.usd_path == (
        "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda"
    )
    for actuator_cfg in robot.actuators.values():
        assert actuator_cfg.effort_limit_sim is None
        assert actuator_cfg.velocity_limit_sim is None
        assert actuator_cfg.stiffness is None
        assert actuator_cfg.damping is None
        assert actuator_cfg.armature is None

    assert pour_env_cfg.FRANKA_PANDA_CFG.spawn.usd_path.endswith("/panda_instanceable.usd")
    assert pour_env_cfg.FRANKA_PANDA_CFG.actuators["panda_shoulder"].stiffness == pytest.approx(80.0)


def test_robot_enables_only_authored_arm_collision_proxies_and_self_collision():
    robot = FrankaPourEnvCfg().scene.robot

    assert robot.spawn.func is pour_env_cfg.spawn_franka_with_arm_collisions
    assert {
        "link0_c",
        "link1_c",
        "link2_c",
        "link3_c",
        "link4_c",
        "link5_c0",
        "link5_c1",
        "link5_c2",
        "link6_c",
        "link7_c",
    } == pour_env_cfg.FRANKA_POUR_ARM_COLLISION_PROXIES
    assert robot.spawn.articulation_props.enabled_self_collisions is True


def test_public_fixed_size_tuple_annotations_are_specific():
    expected = {
        "arm_home": "tuple[float, float, float, float, float, float, float]",
        "curriculum_pour_arm_q": "tuple[float, float, float, float, float, float, float]",
        "curriculum_carry_arm_q": "tuple[float, float, float, float, float, float, float]",
        "tcp_offset_pos": "tuple[float, float, float]",
        "tcp_offset_rot": "tuple[float, float, float, float]",
        "cup_grasp_box_half": "tuple[float, float, float]",
        "cup_grasp_tcp_quat_c": "tuple[float, float, float, float]",
        "cup_reset_pos": "tuple[float, float, float]",
        "target_cup_reset_pos": "tuple[float, float, float]",
        "curriculum_randomized_source_radius_range": "tuple[float, float] | None",
        "curriculum_randomized_reset_tcp_offset_lower": "tuple[float, float, float] | None",
        "curriculum_randomized_reset_tcp_offset_upper": "tuple[float, float, float] | None",
        "particle_workspace_lower_bound": "tuple[float, float, float]",
        "particle_workspace_upper_bound": "tuple[float, float, float]",
    }

    assert {name: FrankaPourEnvCfg.__annotations__[name] for name in expected} == expected


def test_cfg_routes_each_body_to_exactly_one_solver():
    cfg = FrankaPourEnvCfg().finalize()
    solver = cfg.sim.physics.solver_cfg
    entries = {entry.name: entry for entry in solver.entries}
    assert set(entries) == {"arm", "media"}
    assert isinstance(solver, CouplerProxyCfg)

    arm = entries["arm"]
    media = entries["media"]
    assert arm.solver_cfg.integrator == "implicitfast"
    assert arm.solver_cfg.use_mujoco_contacts is False
    assert solver.scene_cfg is cfg.scene
    assert cfg.sim.physics.collision_cfg.soft_contact_max == 0
    assert arm.include_static_shapes is True
    assert arm.bodies == [SceneEntityCfg("robot"), SceneEntityCfg("source_cup"), SceneEntityCfg("target_cup")]
    assert cfg.sim.physics.num_substeps == cfg.physics_substeps == 1
    assert arm.substeps == cfg.rigid_entry_substeps == 4
    assert media.substeps == cfg.mpm_entry_substeps == 2
    assert cfg.sim.physics.num_substeps * media.substeps == 2
    assert media.all_particles is True
    assert media.bodies == [r".*/SpillFloor$"]
    assert media.include_static_shapes is False
    assert media.in_place is True
    assert media.solver_cfg.grid_type == "sparse"
    assert media.solver_cfg.grid_padding == 0
    assert media.solver_cfg.max_active_cell_count == 2 * 512
    assert media.solver_cfg.separate_worlds is True
    assert media.solver_cfg.solver == "jacobi"
    assert media.solver_cfg.warmstart_mode == "none"
    assert media.solver_cfg.max_iterations == 24
    assert cfg.sim.physics.use_cuda_graph is True

    proxies = solver.proxies
    assert len(proxies) == 1
    assert proxies[0].source == "arm" and proxies[0].destination == "media"
    assert proxies[0].bodies == [SceneEntityCfg("source_cup"), SceneEntityCfg("target_cup")]
    assert proxies[0].collision_pipeline is not None
    assert proxies[0].collision_pipeline(None) is None
    assert proxies[0].mass_scale == pytest.approx(cfg.proxy_mass_scale)
    assert cfg.proxy_mass_scale == pytest.approx(100.0)
    assert solver.iterations == cfg.proxy_iterations
    assert not hasattr(pour_env_cfg, "CUP_LABEL_PATTERN")
    assert all(
        "Cup$" not in selector for entry in solver.entries for selector in entry.bodies if isinstance(selector, str)
    )


def test_finalize_propagates_post_init_solver_overrides():
    cfg = FrankaPourEnvCfg()
    cfg.mpm_iterations = 48
    cfg.voxel_size = 0.02
    cfg.physics_substeps = 3
    cfg.rigid_entry_substeps = 4
    cfg.mpm_entry_substeps = 2
    cfg.use_cuda_graph = False
    cfg.proxy_iterations = 4
    cfg.proxy_mass_scale = 0.25

    # Match Hydra's override timing: the nested tree still contains values authored during
    # ``__post_init__`` until finalization resolves the public top-level controls.
    assert _media_entry(cfg).solver_cfg.max_iterations == 24
    assert _media_entry(cfg).solver_cfg.voxel_size == pytest.approx(0.01)

    resolved = cfg.finalize()
    coupled_cfg = resolved.sim.physics.solver_cfg
    entries = {entry.name: entry for entry in coupled_cfg.entries}
    proxy = coupled_cfg.proxies[0]

    assert entries["media"].solver_cfg.max_iterations == 48
    assert entries["media"].solver_cfg.voxel_size == pytest.approx(0.02)
    assert entries["arm"].substeps == 4
    assert entries["media"].substeps == 2
    assert resolved.sim.physics.num_substeps == 3
    assert resolved.sim.physics.use_cuda_graph is False
    assert isinstance(coupled_cfg, CouplerProxyCfg)
    assert coupled_cfg.iterations == 4
    assert proxy.mass_scale == pytest.approx(0.25)


@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("physics_substeps", 0, ValueError),
        ("rigid_entry_substeps", True, ValueError),
        ("mpm_entry_substeps", -1, ValueError),
        ("mpm_iterations", 0, ValueError),
        ("proxy_iterations", 0, ValueError),
        ("voxel_size", 0.0, ValueError),
        ("proxy_mass_scale", float("inf"), ValueError),
        ("use_cuda_graph", 1, TypeError),
    ],
)
def test_solver_controls_reject_invalid_overrides(field, value, error_type):
    cfg = FrankaPourEnvCfg()
    setattr(cfg, field, value)

    with pytest.raises(error_type, match=field):
        cfg.finalize()


def test_visible_source_geometry_matches_grasp_proxy_and_fits_gripper():
    cfg = FrankaPourEnvCfg()
    outer_width = cfg.source_cup_inner_width + 2.0 * cfg.source_cup_wall_thickness
    outer_depth = cfg.source_cup_inner_depth + 2.0 * cfg.source_cup_wall_thickness
    outer_height = cfg.source_cup_cavity_depth + cfg.source_cup_bottom_thickness
    assert (outer_width, outer_depth, outer_height) == pytest.approx((0.056, 0.056, 0.119))
    assert cfg.cup_grasp_box_half == pytest.approx((outer_width / 2.0, outer_depth / 2.0, outer_height / 2.0))
    assert outer_depth < 2.0 * cfg.gripper_open_pos
    assert cfg.grasp_contact_ke >= 1.0e5
    assert cfg.grasp_contact_kd >= 5.0e2
    assert cfg.cup_grasp_box_friction >= 2.0
    # The horizontal fingertip TCP sits well below the rim, so the complete finger pads engage the
    # side wall and the glass rotates around an upper-middle pivot rather than its top edge.
    assert cfg.source_cup_bottom_thickness < cfg.cup_grasp_height < outer_height
    assert cfg.cup_grasp_height == pytest.approx(0.083)
    assert outer_height - cfg.cup_grasp_height == pytest.approx(0.036)
    assert 0.65 * outer_height < cfg.cup_grasp_height < 0.75 * outer_height
    assert cfg.media_fill_frac * cfg.source_cup_cavity_depth == pytest.approx(0.70 * 0.027)
    assert cfg.cup_grasp_tcp_quat_c == pytest.approx((0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)))
    qx, qy, qz, qw = cfg.cup_grasp_tcp_quat_c
    tool_axis_c = (
        2.0 * (qx * qz + qy * qw),
        2.0 * (qy * qz - qx * qw),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    assert tool_axis_c == pytest.approx((1.0, 0.0, 0.0))
    jaw_axis_z = 2.0 * (qy * qz + qx * qw)
    assert jaw_axis_z == pytest.approx(0.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_cup_cavity_depth", 0.0),
        ("source_cup_wall_thickness", float("inf")),
        ("cup_grasp_box_half", (0.028, 0.028, 0.018)),
        ("cup_grasp_height", 0.009),
        ("cup_grasp_height", 0.119),
        ("cup_grasp_tcp_quat_c", (0.0, 0.0, 0.0)),
        ("cup_grasp_tcp_quat_c", (0.0, 0.0, 0.0, 1.1)),
        ("media_fill_frac", 0.0),
        ("media_fill_frac", 1.01),
    ],
)
def test_source_cup_rejects_inconsistent_geometry(field, value):
    cfg = FrankaPourEnvCfg()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        cfg.finalize()


def test_scene_cups_use_narrow_solver_only_builder_hook():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"_build_custom_proto", "_add_cup_body", "_add_target_cup_bodies"}.isdisjoint(function_names)

    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_add_pour_world_to_builder"
    )
    hook_calls = {
        node.func.attr for node in ast.walk(hook) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"_find_world_body", "_add_particle_collider", "_add_rigid_collider"} <= hook_calls

    target_bridge = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_add_kinematic_rigid_object_articulation"
    )
    bridge_calls = {
        node.func.attr
        for node in ast.walk(target_bridge)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"add_joint_free", "add_articulation"} <= bridge_calls


def test_randomized_ik_branch_ranking_prefers_open_joint6_complete_paths_and_falls_back():
    """Open reset branches must be preferred before ranking complete receiver paths."""
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    build_bank = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_randomized_reset_bank"
    )
    row_has_open_branch = next(
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "row_has_open_reset_branch" for target in node.targets)
    )
    open_path_filter = next(
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "complete_path_valid"
        and node.lineno > row_has_open_branch.lineno
    )
    collision_path_indices = next(
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "collision_path_indices" for target in node.targets)
    )
    global_path_indices = next(
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "global_path_indices" for target in node.targets)
    )

    assert ast.unparse(row_has_open_branch.value) == (
        "(complete_path_valid & open_reset_branch[:, :, None, None]).any(dim=(1, 2, 3), keepdim=True)"
    )
    assert ast.unparse(open_path_filter.value) == ("~row_has_open_reset_branch | open_reset_branch[:, :, None, None]")
    assert isinstance(global_path_indices.value, ast.Attribute)
    assert ast.unparse(global_path_indices.value.value.args[0]) == "flat_complete_path_score"
    assert ast.unparse(collision_path_indices.value) == (
        "torch.cat((deterministic_source_path_indices, global_path_indices), dim=-1)"
    )
    assert row_has_open_branch.lineno < open_path_filter.lineno < collision_path_indices.lineno

    complete_path_valid = torch.tensor(
        [
            [[[True, False]], [[True, True]], [[False, False]]],
            [[[True, True]], [[False, False]], [[True, False]]],
        ]
    )
    open_reset_branch = torch.tensor([[False, True, True], [False, True, False]])
    row_has_open_reset_branch = (complete_path_valid & open_reset_branch[:, :, None, None]).any(
        dim=(1, 2, 3), keepdim=True
    )
    ranking_mask = ~row_has_open_reset_branch | open_reset_branch[:, :, None, None]

    # Row 0 ranks only complete paths rooted at its available open reset branch. Row 1 has no
    # complete open branch, so both valid folded reset branches remain available as fallbacks.
    assert (complete_path_valid & ranking_mask).flatten(start_dim=1).sum(dim=-1).tolist() == [2, 3]
    assert (complete_path_valid & ranking_mask)[0, 0].sum() == 0
    assert (complete_path_valid & ranking_mask)[0, 1].sum() == 2


def test_invalid_initial_ik_rows_are_sanitized_then_filtered_from_reset_pools():
    """An infeasible reset row must not poison batched continuation IK or become sampleable."""
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    build_bank = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_randomized_reset_bank"
    )

    def first_assignment(name: str) -> ast.Assign:
        return min(
            (
                node
                for node in ast.walk(build_bank)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            ),
            key=lambda node: node.lineno,
        )

    reset_has_candidate = first_assignment("reset_has_candidate")
    fallback_indices = first_assignment("fallback_indices")
    trajectory_q = first_assignment("trajectory_q")
    randomized_collision_free = first_assignment("randomized_collision_free")
    assert ast.unparse(reset_has_candidate.value) == "candidate_valid.any(dim=-1)"
    assert ast.unparse(fallback_indices.value) == (
        "torch.where(reset_has_candidate, valid_fallback_indices, margin_fallback_indices)"
    )
    assert ast.unparse(trajectory_q.value) == (
        "torch.where(trajectory_valid.unsqueeze(-1), expanded_q, fallback_q).clone()"
    )
    assert "row_has_collision_free_path[randomized_rows]" in ast.unparse(randomized_collision_free.value)

    candidate_valid = torch.tensor([[False, True, False], [False, False, False]])
    expanded_margin = torch.tensor([[0.1, 0.2, 0.3], [float("nan"), -0.4, 0.5]])
    expanded_q = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    reset_has_candidate_t = candidate_valid.any(dim=-1)
    valid_fallback_indices = torch.argmax(candidate_valid.to(dtype=torch.int32), dim=-1)
    margin_fallback_indices = torch.argmax(torch.nan_to_num(expanded_margin, nan=-torch.inf), dim=-1)
    fallback_indices_t = torch.where(
        reset_has_candidate_t,
        valid_fallback_indices,
        margin_fallback_indices,
    )
    fallback_q = expanded_q[torch.arange(2), fallback_indices_t].unsqueeze(1)
    sanitized = torch.where(candidate_valid.unsqueeze(-1), expanded_q, fallback_q)

    assert fallback_indices_t.tolist() == [1, 2]
    torch.testing.assert_close(sanitized[:, :, 0], torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]))
    # Sanitization supplies finite continuation seeds without changing validity; the second row
    # therefore remains ineligible for the eventual reset-index pool.
    assert candidate_valid.any(dim=-1).tolist() == [True, False]


def test_randomized_collision_screen_covers_full_path_with_authored_finger_states():
    """Collision validation must include pour/tilt with the fingers preloaded around the glass."""
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    build_bank = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_randomized_reset_bank"
    )
    collision_call = next(
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_collision_free_ik_candidates"
    )
    source_candidates_assignment = next(
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "source_collision_candidates" for target in node.targets)
    )
    collision_waypoints = collision_call.args[1]
    assert isinstance(collision_waypoints, ast.Tuple)
    assert isinstance(collision_waypoints.elts[0], ast.Starred)
    assert ast.unparse(collision_waypoints.elts[0].value) == "source_collision_candidates"
    assert isinstance(source_candidates_assignment.value, ast.Tuple)
    source_waypoints = source_candidates_assignment.value.elts
    assert all(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "gather_source_waypoint"
        for call in source_waypoints
    )
    assert [ast.unparse(call.args[0]) for call in source_waypoints] == [
        "expanded_q",
        "pregrasp_candidates",
        "midgrasp_candidates",
        "grasp_candidates",
        "carry_candidates",
    ]
    assert [ast.unparse(call.args[1]) for call in source_waypoints] == [
        "float(self.cfg.gripper_open_pos)",
        "float(self.cfg.gripper_open_pos)",
        "float(self.cfg.gripper_open_pos)",
        "float(self.cfg.gripper_open_pos)",
        "float(self.cfg.gripper_preload_pos)",
    ]
    assert [ast.unparse(waypoint) for waypoint in collision_waypoints.elts[1:]] == [
        "pour_collision_candidates",
        "tilt_collision_candidates",
    ]
    collision_candidate_source = ast.unparse(build_bank)
    assert (
        "pour_collision_candidates[:, :, finger_coordinate_ids] = float(self.cfg.gripper_preload_pos)"
        in collision_candidate_source
    )
    assert (
        "tilt_collision_candidates[:, :, finger_coordinate_ids] = float(self.cfg.gripper_preload_pos)"
        in collision_candidate_source
    )


def test_randomized_reset_bank_builds_ik_targets_in_environment_local_frame():
    """The reset bank must not inherit the clone layout's batch-dependent world offset."""
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    build_bank = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_randomized_reset_bank"
    )

    assignments = {
        target.id: node.value
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert ast.unparse(assignments["prototype_origin"]) == "-self.env_origins[0]"
    assert ast.unparse(assignments["target_positions_w"]) == "tcp_positions"
    assert ast.unparse(assignments["pregrasp_target_positions_w"]) == "pregrasp_tcp_positions"
    assert ast.unparse(assignments["grasp_target_positions_w"]) == "grasp_tcp_positions"

    # One fresh centered builder feeds collision validation and another feeds the local IK model.
    local_builder_calls = [
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "local_prototype_builder"
    ]
    assert len(local_builder_calls) == 2
    env_origin_references = [
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Subscript) and ast.unparse(node) == "self.env_origins[0]"
    ]
    assert len(env_origin_references) == 1

    collision_validation_call = next(
        node
        for node in ast.walk(build_bank)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_collision_free_ik_candidates"
    )
    assert ast.unparse(collision_validation_call.args[0]) == "prototype_builder"


def test_polar_source_workspace_defaults_cover_a_broad_base_centered_sector():
    """The preset's polar domain must be broad and fit its conservative Cartesian bounds."""
    cfg = FrankaPourEnvCfg()

    assert cfg.curriculum_randomized_source_radius_range == pytest.approx((0.40, 0.78))
    assert cfg.curriculum_randomized_source_azimuth_range == pytest.approx(math.radians(35.0))
    assert cfg.curriculum_randomized_source_xy_correlation == pytest.approx(0.0)

    minimum_radius, maximum_radius = cfg.curriculum_randomized_source_radius_range
    azimuth = cfg.curriculum_randomized_source_azimuth_range
    required_half_range = (
        max(
            abs(minimum_radius * math.cos(azimuth) - cfg.cup_reset_pos[0]),
            abs(maximum_radius - cfg.cup_reset_pos[0]),
        ),
        maximum_radius * math.sin(azimuth),
    )
    assert all(
        configured >= required
        for configured, required in zip(
            cfg.curriculum_randomized_source_position_range,
            required_half_range,
            strict=True,
        )
    )


def test_builder_hook_limits_lookups_to_current_world_tail():
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv

    builder = SimpleNamespace(
        body_world=[None, None, 0, 0, 1, 1, 1],
        shape_world=[None, 0, 0, 1, 1],
    )

    assert FrankaPourEnv._current_world_range(builder, "body", 1) == range(4, 7)
    assert FrankaPourEnv._current_world_range(builder, "shape", 1) == range(3, 5)
    with pytest.raises(RuntimeError, match="no body entries for open world 2"):
        FrankaPourEnv._current_world_range(builder, "body", 2)


def test_task_has_receiving_cup_safe_actions_and_success_threshold():
    cfg = FrankaPourEnvCfg()
    assert cfg.target_cup_reset_pos != cfg.cup_reset_pos
    assert isinstance(cfg.actions.arm_action, mdp.RelativeJointPositionActionCfg)
    assert cfg.actions.arm_action.joint_names == [f"panda_joint{i}" for i in range(1, 8)]
    assert cfg.actions.arm_action.preserve_order is True
    assert cfg.actions.arm_action.scale == pytest.approx(0.08)
    assert cfg.actions.arm_action.use_zero_offset is True
    assert not hasattr(cfg.actions.arm_action, "waypoint_phases")
    # Keep a meaningful particle margin below the demonstrated trajectory's lower tail while the
    # one-time delivery reward still drives transfer beyond the success threshold.
    assert cfg.curriculum_target_frac[-1] == pytest.approx(0.30)
    assert cfg.episode_length_s == pytest.approx(5.0)


def test_training_curriculum_closes_by_default_and_penalizes_irrecoverable_spill():
    cfg = FrankaPourEnvCfg()
    assert type(cfg.actions.gripper_action).__name__ == "CurriculumGripperPositionActionCfg"
    assert cfg.actions.gripper_action.scale == pytest.approx(0.016)
    assert cfg.actions.gripper_action.alpha == pytest.approx(0.2)
    assert cfg.actions.gripper_action.use_incremental_target is False
    assert cfg.actions.gripper_action.default_position == pytest.approx(cfg.gripper_preload_pos)
    assert cfg.actions.gripper_action.neutral_position == pytest.approx(cfg.gripper_open_pos)
    assert cfg.actions.gripper_action.limit_to_preload is False
    assert cfg.actions.gripper_action.force_open_before_phase_stage == -1
    assert not hasattr(cfg.actions.gripper_action, "hold_offset_through_stage")
    assert cfg.actions.gripper_action.close_position == pytest.approx(0.021)
    assert cfg.actions.gripper_action.open_position == pytest.approx(cfg.gripper_open_pos)
    reward_names = {name for name, term_cfg in vars(cfg.rewards).items() if isinstance(term_cfg, RewardTermCfg)}
    assert reward_names == {
        "task_progress",
        "approach_progress",
        "grasp_lift_progress",
        "delivered",
        "success",
        "spill",
        "failure",
        "action_rate",
        "action_magnitude",
    }
    assert cfg.rewards.task_progress.func is mdp.PourTaskProgress
    assert cfg.rewards.task_progress.weight == pytest.approx(5.0)
    assert cfg.rewards.task_progress.params["grasp_reach_std"] == pytest.approx(0.015)
    assert cfg.rewards.task_progress.params["grasp_preload_position"] == pytest.approx(0.024)
    assert cfg.rewards.task_progress.params["source_offset_xy"] == pytest.approx(cfg.pour_source_offset_xy)
    assert cfg.rewards.task_progress.params["target_tilt"] == pytest.approx(math.radians(140.0))
    minimum_drain_tilt = math.atan2(cfg.source_cup_cavity_depth, 0.5 * cfg.source_cup_inner_depth)
    assert cfg.rewards.task_progress.params["target_tilt"] > minimum_drain_tilt
    assert cfg.rewards.task_progress.params["target_tilt"] < math.pi
    assert cfg.rewards.task_progress.params["pour_direction_xy"] == pytest.approx((0.0, -1.0))
    assert cfg.rewards.task_progress.params["source_mouth_height"] == pytest.approx(
        cfg.source_cup_bottom_thickness + cfg.source_cup_cavity_depth
    )
    assert cfg.rewards.task_progress.params["alignment_radius"] == pytest.approx(0.15)
    assert cfg.rewards.task_progress.params["active_through_stage"] == cfg.curriculum_stage_names.index("carry")
    assert cfg.rewards.task_progress.params["min_lift_height"] == pytest.approx(0.05)
    assert cfg.rewards.task_progress.params["max_tcp_distance"] == pytest.approx(0.018)
    assert cfg.rewards.task_progress.params["max_gripper_width_error"] == pytest.approx(0.006)
    assert cfg.rewards.task_progress.params["max_gripper_command"] == pytest.approx(
        cfg._resolved_success_max_gripper_command()
    )
    assert not hasattr(cfg.rewards, "reach")
    assert not hasattr(cfg.rewards, "grasp")
    assert not hasattr(cfg.rewards, "lift")
    assert not hasattr(cfg.rewards, "align")
    assert not hasattr(cfg.rewards, "tilt")
    assert cfg.rewards.approach_progress.func is mdp.ApproachProgress
    assert cfg.rewards.approach_progress.weight == pytest.approx(8.0)
    assert cfg.rewards.approach_progress.params["position_std"] == pytest.approx(0.20)
    assert cfg.rewards.approach_progress.params["orientation_std"] == pytest.approx(0.75)
    assert cfg.rewards.approach_progress.params["open_hand_fraction"] == pytest.approx(0.35)
    assert cfg.rewards.approach_progress.params["active_from_stage"] == cfg.curriculum_stage_names.index("approach_1")
    assert cfg.rewards.grasp_lift_progress.func is mdp.GraspLiftProgress
    assert cfg.rewards.grasp_lift_progress.weight == pytest.approx(10.0)
    assert cfg.rewards.grasp_lift_progress.params["target_height"] == pytest.approx(0.10)
    assert cfg.rewards.grasp_lift_progress.params["grasp_reach_std"] == pytest.approx(0.025)
    assert cfg.rewards.grasp_lift_progress.params["grasp_fraction"] == pytest.approx(0.40)
    assert cfg.rewards.grasp_lift_progress.params["active_from_stage"] == cfg.curriculum_stage_names.index("near_carry")
    assert cfg.rewards.delivered.func is mdp.HeldDeliveryProgress
    assert cfg.rewards.delivered.params["min_lift_height"] == pytest.approx(0.05)
    assert cfg.rewards.delivered.params["max_tcp_distance"] == pytest.approx(0.018)
    assert cfg.rewards.delivered.params["max_gripper_width_error"] == pytest.approx(0.006)
    assert cfg.rewards.delivered.params["max_gripper_command"] == pytest.approx(
        cfg._resolved_success_max_gripper_command()
    )
    assert cfg.rewards.spill.func is mdp.NewlySpilledParticles
    assert cfg.rewards.task_progress.weight < cfg.rewards.success.weight
    assert cfg.rewards.spill.weight == pytest.approx(-30.0)
    assert cfg.rewards.failure.func is mdp.terminal_failure
    assert cfg.rewards.failure.weight == pytest.approx(-35.0)
    assert cfg.terminations.spill.func is mdp.excessive_spill
    assert cfg.terminations.extreme_rigid_state.func is mdp.extreme_rigid_state
    assert cfg.terminations.lost_grasp.func is mdp.lost_lifted_grasp
    assert cfg.terminations.lost_grasp.params["dwell_time_s"] == pytest.approx(0.05)
    assert cfg.terminations.success.func is mdp.stable_pour_success
    assert cfg.terminations.success.params["dwell_time_s"] == pytest.approx(0.15)
    assert cfg.terminations.success.params["min_lift_height"] == pytest.approx(0.05)
    assert cfg.terminations.success.params["max_tcp_distance"] == pytest.approx(0.018)
    assert cfg.terminations.success.params["max_gripper_width_error"] == pytest.approx(0.006)
    assert cfg.terminations.success.params["max_gripper_command"] == pytest.approx(
        cfg._resolved_success_max_gripper_command()
    )
    termination_names = [
        name for name, term_cfg in vars(cfg.terminations).items() if isinstance(term_cfg, TerminationTermCfg)
    ]
    assert termination_names[-2:] == ["success", "time_out"]
    assert cfg.terminations.time_out.func is mdp.unsuccessful_time_out
    assert cfg.max_spill_fraction == pytest.approx(0.10)
    assert cfg.spill_table_height == pytest.approx(0.0)
    assert cfg.state_bound_joint_position_margin == pytest.approx(0.05)
    assert cfg.state_bound_max_joint_velocity == pytest.approx(20.0)
    assert cfg.state_bound_max_cup_linear_velocity == pytest.approx(10.0)
    assert cfg.state_bound_max_cup_angular_velocity == pytest.approx(50.0)
    assert abs(cfg.rewards.action_rate.weight) <= 0.002
    assert cfg.rewards.action_magnitude.func is mdp.action_l2
    assert cfg.rewards.action_magnitude.weight == pytest.approx(-0.05)

    agent = FrankaPourPPORunnerCfg()
    assert agent.class_name == "OnPolicyRunner"
    assert agent.save_interval == 50
    assert agent.actor.distribution_cfg.class_name == "GaussianDistribution"
    assert agent.actor.distribution_cfg.init_std == pytest.approx(0.1)
    assert agent.actor.distribution_cfg.std_type == "log"
    assert agent.actor.obs_normalization is False
    assert agent.critic.obs_normalization is False
    assert agent.clip_actions == pytest.approx(1.0)
    assert agent.algorithm.entropy_coef == pytest.approx(1.0e-3)
    assert agent.algorithm.num_learning_epochs == 5
    assert agent.algorithm.clip_param == pytest.approx(0.2)
    assert agent.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert agent.algorithm.max_grad_norm == pytest.approx(1.0)
    assert agent.algorithm.schedule == "fixed"
    assert cfg.rewards.task_progress.params["discount_factor"] == pytest.approx(agent.algorithm.gamma)
    assert cfg.rewards.approach_progress.params["discount_factor"] == pytest.approx(agent.algorithm.gamma)
    assert cfg.rewards.grasp_lift_progress.params["discount_factor"] == pytest.approx(agent.algorithm.gamma)
    assert agent.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    assert agent.logger == "wandb"
    assert agent.wandb_project == "franka-pour-mpm"


def test_backward_curriculum_config_is_complete_and_play_uses_randomized_task():
    cfg = FrankaPourEnvCfg()
    stage_count = len(cfg.curriculum_stage_names)

    assert cfg.is_finite_horizon is True
    assert isinstance(cfg.curriculum.stage, CurriculumTermCfg)
    assert cfg.curriculum.stage.func is mdp.PourCurriculum
    assert not hasattr(cfg, "reset_mixture_probabilities")
    assert cfg.curriculum_stage_names == (
        "drain",
        "deep_tilt",
        "tilt",
        "pour",
        "near_carry",
        "mid_carry",
        "carry",
        "grasp",
        "approach_1",
        "approach_2",
        "approach_3",
        "approach_4",
        "approach_5",
        "approach_6",
        "full",
        "randomized",
    )
    assert stage_count == 16
    assert len(cfg.curriculum_target_frac) == stage_count
    assert cfg.curriculum_target_frac == pytest.approx(
        (
            0.05,
            0.08,
            0.10,
            0.15,
            0.15,
            0.18,
            0.20,
            0.30,
            0.30,
            0.30,
            0.30,
            0.30,
            0.30,
            0.30,
            0.30,
            0.30,
        )
    )
    assert list(cfg.curriculum_target_frac) == sorted(cfg.curriculum_target_frac)
    assert cfg.curriculum_start_stage == 0
    assert cfg.curriculum_freeze is False
    assert cfg.curriculum_min_resets_per_stage == 4096
    assert cfg.curriculum_min_reset_cohorts_per_stage == pytest.approx(8.0)
    assert cfg.curriculum_previous_stage_replay_fraction == pytest.approx(0.1)
    assert cfg.curriculum_frontier_entry_replay_fraction == pytest.approx(0.5)
    assert cfg.curriculum_transport_reset_fractions == pytest.approx((1.0 / 3.0, 2.0 / 3.0))
    assert cfg.curriculum_grasp_approach_fractions == pytest.approx((0.75, 0.50, 0.375, 0.25, 0.125, 0.0))
    transport_arm_q = cfg._curriculum_transport_arm_configs()
    assert len(transport_arm_q) == 2
    for fraction, arm_q in zip(cfg.curriculum_transport_reset_fractions, transport_arm_q, strict=True):
        assert arm_q == pytest.approx(
            tuple(
                (1.0 - fraction) * pour_q + fraction * carry_q
                for pour_q, carry_q in zip(cfg.curriculum_pour_arm_q, cfg.curriculum_carry_arm_q, strict=True)
            )
        )
    assert cfg.curriculum_randomization_extent_levels == pytest.approx(
        (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0)
    )
    assert cfg.curriculum_independent_arm_fraction_levels == pytest.approx(
        (0.0, 0.0, 0.0, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
    )
    assert cfg.curriculum_independent_target_fraction_levels == pytest.approx(
        (0.0, 0.0, 0.0, 0.10, 0.25, 0.50, 0.70, 0.85, 1.0)
    )
    assert stage_count - 1 + len(cfg.curriculum_randomization_extent_levels) == 24
    assert cfg.curriculum_randomization_promotion_threshold == pytest.approx(0.65)
    assert cfg.curriculum_randomization_start_level == 0
    # Do not expose a generic stage identifier. The actor does receive behaviorally relevant
    # finite-state variables and the active delivery goal so the finite-horizon MDP is Markov.
    assert not hasattr(cfg.observations.policy, "curriculum_context")
    assert not hasattr(cfg.observations.policy, "arm_reference_phase")
    assert not hasattr(cfg.observations.policy, "arm_reference_error")
    assert not hasattr(cfg.observations.policy, "trajectory_status")
    assert cfg.observations.policy.time_remaining.func is mdp.time_remaining_obs
    assert cfg.observations.policy.pour_target_fraction.func is mdp.pour_target_fraction_obs
    assert not hasattr(cfg.observations.policy, "success_dwell")
    assert not hasattr(cfg.observations.policy, "lost_grasp_dwell")
    assert cfg.observations.policy.target_pose.func is mdp.target_pose_obs
    assert not hasattr(cfg.observations.policy, "particle_fractions")
    assert not hasattr(cfg.observations.policy, "particle_transfer")
    assert cfg.observations.policy.arm_q.scale == pytest.approx(0.3)
    assert cfg.observations.policy.arm_qd.scale == pytest.approx(0.05)
    assert cfg.observations.policy.tcp_to_grasp_position_c.func is mdp.tcp_to_grasp_position_c_obs
    assert cfg.observations.policy.tcp_to_grasp_position_c.scale == pytest.approx(10.0)
    assert cfg.observations.policy.grasp_to_tcp_quat.func is mdp.grasp_to_tcp_quat_obs
    assert cfg.observations.policy.target_position_c.func is mdp.target_position_c_obs
    assert cfg.observations.policy.target_position_c.scale == pytest.approx(5.0)
    assert cfg.observations.policy.finger_position.func is mdp.finger_position_obs
    assert cfg.observations.policy.finger_position.scale == pytest.approx(25.0)
    assert cfg.observations.policy.finger_velocity.func is mdp.finger_velocity_obs
    assert cfg.observations.policy.finger_velocity.scale == pytest.approx(5.0)
    assert cfg.observations.policy.last_action.scale == pytest.approx(0.2)
    assert cfg.observations.policy.gripper_contact.func is mdp.gripper_contact_obs
    assert cfg.observations.policy.gripper_contact.scale == pytest.approx(250.0)
    assert cfg.observations.privileged.success_dwell.func is mdp.success_dwell_obs
    assert cfg.observations.privileged.lost_grasp_dwell.func is mdp.lost_grasp_dwell_obs
    assert cfg.observations.privileged.cup_velocity.func is mdp.cup_velocity_obs
    assert cfg.observations.privileged.particle_fractions.func is mdp.particle_fractions_obs
    assert cfg.observations.privileged.particle_transfer.func is mdp.particle_transfer_obs
    assert cfg.observations.privileged.held_delivery_history.func is mdp.held_delivery_history_obs
    assert cfg.actions.gripper_action.contact_min_deflection == pytest.approx(0.001)
    assert cfg.actions.gripper_action.contact_max_velocity == pytest.approx(0.05)

    agent = FrankaPourPPORunnerCfg()
    episode_steps = round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    documented_training_env_count = 512
    available_resets = documented_training_env_count * agent.max_iterations * agent.num_steps_per_env // episode_steps
    resets_per_frontier = max(
        cfg.curriculum_min_resets_per_stage,
        math.ceil(cfg.curriculum_min_reset_cohorts_per_stage * documented_training_env_count),
    )
    required_resets = resets_per_frontier * (stage_count - 1 + len(cfg.curriculum_randomization_extent_levels))
    assert required_resets <= available_resets

    assert cfg.curriculum_randomized_source_position_range == pytest.approx((0.30, 0.45))
    assert cfg.curriculum_randomized_source_radius_range == pytest.approx((0.40, 0.78))
    assert cfg.curriculum_randomized_source_azimuth_range == pytest.approx(math.radians(35.0))
    assert cfg.curriculum_randomized_source_xy_correlation == pytest.approx(0.0)
    assert cfg.curriculum_randomized_carry_position_range == pytest.approx((0.03, 0.03))
    assert cfg.curriculum_randomized_source_yaw_range == pytest.approx(math.radians(30.0))
    assert cfg.curriculum_randomized_target_center_xy == pytest.approx((0.50, -0.18))
    assert cfg.curriculum_randomized_target_position_range == pytest.approx((0.15, 0.44))
    assert cfg.curriculum_randomized_cup_clearance == pytest.approx(0.04)
    assert cfg.curriculum_grasp_descent_overshoot == pytest.approx(0.0)
    assert cfg.curriculum_randomized_reset_tcp_standoff == pytest.approx((-0.12, 0.0, 0.0))
    assert cfg.curriculum_randomized_reset_tcp_jitter == pytest.approx((0.02, 0.03, 0.0))
    assert cfg.curriculum_randomized_reset_tcp_offset_lower == pytest.approx((-0.16, -0.16, 0.0))
    assert cfg.curriculum_randomized_reset_tcp_offset_upper == pytest.approx((0.02, 0.16, 0.25))
    assert cfg.curriculum_randomized_reset_tcp_rotation_angle_range == pytest.approx(
        (math.radians(20.0), math.radians(60.0))
    )
    assert cfg.curriculum_randomized_reset_tcp_min_grasp_distance == pytest.approx(0.09)
    assert cfg.curriculum_randomized_pour_clearance == pytest.approx(0.01)
    assert cfg.curriculum_randomized_reset_ik_grid_size == 7
    assert cfg.curriculum_randomized_reset_ik_samples_per_source == 11
    assert cfg.curriculum_randomized_min_source_cell_fraction == pytest.approx(0.2)
    assert cfg.curriculum_randomized_min_reset_variants_per_source == 2
    assert cfg.curriculum_randomized_reset_joint6_max == pytest.approx(3.75)

    arm_configs = (
        cfg.curriculum_drain_arm_q,
        cfg.curriculum_deep_tilt_arm_q,
        cfg.curriculum_tilt_arm_q,
        cfg.curriculum_pour_arm_q,
        cfg.curriculum_carry_arm_q,
        cfg.arm_home,
        cfg.arm_home,
        cfg.arm_home,
    )
    for arm_q in arm_configs:
        assert len(arm_q) == 7
        for position, (lower, upper) in zip(arm_q, pour_env_cfg.PANDA_ARM_JOINT_LIMITS, strict=True):
            assert lower <= position <= upper

    for preset in (FrankaPourEnvCfg_PLAY(), FrankaPourEnvCfg_TELEOP()):
        assert preset.curriculum_start_stage == stage_count - 1
        assert preset.curriculum_randomization_start_level == len(preset.curriculum_randomization_extent_levels) - 1
        assert preset.curriculum_freeze is True


def test_reset_dataset_config_uses_adaptive_sampling_and_frozen_evaluation():
    cfg = FrankaPourEnvCfg_RESET_DATASET()
    eval_cfg = FrankaPourEnvCfg_RESET_DATASET_EVAL()
    agent = FrankaPourResetDatasetPPORunnerCfg()

    assert cfg.curriculum.reset_dataset.func is mdp.PourResetDatasetCurriculum
    assert cfg.reset_dataset_path == "datasets/franka_pour/reset_dataset.pt"
    assert cfg.reset_dataset_content_sha256 is None
    assert cfg.reset_dataset_top_grasp_count is None
    for legacy_field in (
        "reset_mixture_probabilities",
        "reset_mixture_near_object_open_phase_probabilities",
        "reset_mixture_near_object_preloaded_probability",
        "reset_mixture_statistics_window_size",
        "reset_mixture_validated_cache_path",
        "reset_mixture_validation_steps",
    ):
        assert not hasattr(cfg, legacy_field)
    assert cfg.reset_dataset_sampler.target_success_rate == pytest.approx(0.50)
    assert cfg.reset_dataset_sampler.temperature == pytest.approx(0.10)
    assert cfg.reset_dataset_sampler.history_capacity == 32
    assert cfg.reset_dataset_sampler.prior_strength == pytest.approx(4.0)
    assert cfg.reset_dataset_sampler.initial_frontier_size == 128
    assert cfg.reset_dataset_sampler.probe_size == 256
    assert cfg.reset_dataset_sampler.probe_fraction == pytest.approx(0.10)
    assert cfg.reset_dataset_sampler.replay_fraction == pytest.approx(0.10)
    assert cfg.reset_dataset_sampler.frontier_evidence == pytest.approx(2.0)
    assert cfg.pour_target_frac == pytest.approx(0.30)
    assert isinstance(cfg.actions.arm_action, mdp.RelativeJointPositionActionCfg)
    assert cfg.actions.arm_action.joint_names == [f"panda_joint{i}" for i in range(1, 8)]
    assert cfg.actions.arm_action.preserve_order is True
    assert cfg.actions.arm_action.scale == pytest.approx(0.015)
    assert cfg.actions.arm_action.use_zero_offset is True
    assert cfg.actions.gripper_action.use_incremental_target is False
    assert cfg.actions.gripper_action.binary_threshold == pytest.approx(0.0)
    assert cfg.actions.gripper_action.alpha == pytest.approx(1.0 - 0.8 ** (1.0 / 3.0))
    assert eval_cfg.curriculum_freeze is True
    assert eval_cfg.reset_dataset_path == cfg.reset_dataset_path
    assert eval_cfg.reset_dataset_top_grasp_count is None
    reward_names = {name for name, term_cfg in vars(cfg.rewards).items() if isinstance(term_cfg, RewardTermCfg)}
    assert reward_names == {
        "reach",
        "goal_distance",
        "success",
        "action_magnitude",
        "action_rate",
        "joint_velocity",
        "failure",
    }
    assert cfg.rewards.reach.func is mdp.tcp_cup_distance_tanh
    assert cfg.rewards.reach.weight == pytest.approx(0.1)
    assert cfg.rewards.reach.params["std"] == pytest.approx(0.3)
    assert cfg.rewards.goal_distance.func is mdp.media_target_distance_tanh
    assert cfg.rewards.goal_distance.weight == pytest.approx(0.1)
    assert cfg.rewards.goal_distance.params["std"] == pytest.approx(0.2)
    assert cfg.rewards.success.func is mdp.pour_success_bonus
    assert cfg.rewards.success.weight == pytest.approx(1.0)
    assert cfg.rewards.success.params == {}
    assert cfg.rewards.action_magnitude.func is mdp.action_l2
    assert cfg.rewards.action_magnitude.weight == pytest.approx(-1.0e-4)
    assert cfg.rewards.action_rate.func is mdp.action_rate_l2
    assert cfg.rewards.action_rate.weight == pytest.approx(-1.0e-3)
    assert cfg.rewards.joint_velocity.func is mdp.finite_joint_velocity_l2
    assert cfg.rewards.joint_velocity.weight == pytest.approx(-1.0e-2)
    assert cfg.rewards.failure.func is mdp.terminal_failure
    assert cfg.rewards.failure.weight == pytest.approx(-1.0)
    assert cfg.rewards.failure.params == {"include_time_out": False}
    assert cfg.terminations.success.func is mdp.immediate_pour_success
    assert cfg.terminations.success.params == {}
    assert cfg.terminations.lost_grasp.params["terminate"] is False
    assert cfg.terminations.spill.params["terminate"] is True
    assert cfg.max_spill_fraction == pytest.approx(0.30)
    assert cfg.terminations.time_out.func is mdp.unsuccessful_time_out
    assert cfg.terminations.time_out.time_out is True
    assert cfg.episode_length_s == pytest.approx(7.0)
    assert cfg.is_finite_horizon is False
    assert cfg.decimation == 4
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.observations.policy.history_length == 13
    policy_step_s = cfg.sim.dt * cfg.decimation
    assert 1.0 / policy_step_s == pytest.approx(30.0)
    assert agent.num_steps_per_env * policy_step_s == pytest.approx(32.0 / 30.0)
    assert round(cfg.episode_length_s / policy_step_s) == 210
    assert not any(
        "stage" in parameter
        for term_cfg in vars(cfg.rewards).values()
        if isinstance(term_cfg, RewardTermCfg)
        for parameter in (term_cfg.params or {})
    )
    assert vars(eval_cfg.rewards) == vars(cfg.rewards)
    finalized = cfg.finalize()
    assert isinstance(finalized.rewards, pour_env_cfg.ResetDatasetRewardsCfg)
    assert isinstance(finalized.actions.arm_action, mdp.RelativeJointPositionActionCfg)
    assert finalized.actions.arm_action.scale == pytest.approx(0.015)
    assert finalized.actions.gripper_action.binary_threshold == pytest.approx(0.0)

    overridden = FrankaPourEnvCfg_RESET_DATASET()
    overridden.actions.arm_action.scale = 0.015
    overridden.decimation = 6
    overridden = overridden.finalize()
    assert overridden.actions.arm_action.scale == pytest.approx(0.015)
    assert overridden.sim.render_interval == 6
    assert agent.experiment_name == "franka_pour_reset_dataset_joint_rel"
    assert agent.run_name == "reset_dataset_joint_rel"


def test_reset_dataset_play_preserves_policy_abi_with_captured_sparse_grid():
    eval_cfg = FrankaPourEnvCfg_RESET_DATASET_EVAL().finalize()
    play_cfg = FrankaPourEnvCfg_RESET_DATASET_PLAY().finalize()

    assert play_cfg.scene.num_envs == 1
    assert play_cfg.curriculum_freeze is True
    assert play_cfg.reset_dataset_path == eval_cfg.reset_dataset_path
    assert play_cfg.reset_dataset_content_sha256 == eval_cfg.reset_dataset_content_sha256
    assert play_cfg.reset_dataset_top_grasp_count == eval_cfg.reset_dataset_top_grasp_count
    assert play_cfg.actions.to_dict() == eval_cfg.actions.to_dict()
    assert play_cfg.observations.policy.to_dict() == eval_cfg.observations.policy.to_dict()
    assert play_cfg.episode_length_s == pytest.approx(eval_cfg.episode_length_s)
    assert _media_entry(eval_cfg).solver_cfg.grid_type == "sparse"
    assert eval_cfg.sim.physics.use_cuda_graph is True
    play_solver_cfg = _media_entry(play_cfg).solver_cfg
    assert play_solver_cfg.grid_type == "sparse"
    assert play_solver_cfg.grid_padding == 0
    assert play_solver_cfg.max_active_cell_count == 512
    assert play_solver_cfg.separate_worlds is True
    assert play_cfg.sim.physics.use_cuda_graph is True
    assert play_cfg.terminations.success.func is mdp.immediate_pour_success
    assert play_cfg.terminations.success.params == {}
    assert play_cfg.viewer.lookat == pytest.approx(eval_cfg.viewer.lookat)
    assert play_cfg.viewer.eye == pytest.approx((0.9, 0.65, 0.5))
    eval_distance = math.dist(eval_cfg.viewer.eye, eval_cfg.viewer.lookat)
    play_distance = math.dist(play_cfg.viewer.eye, play_cfg.viewer.lookat)
    assert eval_distance - play_distance == pytest.approx(1.0, abs=0.02)
    assert play_cfg.sim.default_visualizer_cfg.eye == pytest.approx(play_cfg.viewer.eye)
    assert play_cfg.sim.default_visualizer_cfg.lookat == pytest.approx(play_cfg.viewer.lookat)


def test_reset_dataset_semantics_do_not_change_traditional_curriculum_defaults():
    reset_dataset = FrankaPourEnvCfg_RESET_DATASET()
    reverse = FrankaPourEnvCfg()

    assert reset_dataset.rewards.success.func is mdp.pour_success_bonus
    assert reset_dataset.rewards.failure.params == {"include_time_out": False}
    assert reset_dataset.terminations.success.func is mdp.immediate_pour_success
    assert reset_dataset.terminations.success.params == {}
    assert reset_dataset.terminations.lost_grasp.params["terminate"] is False
    assert reset_dataset.terminations.spill.params["terminate"] is True
    assert reset_dataset.max_spill_fraction == pytest.approx(0.30)
    assert reset_dataset.terminations.time_out.func is mdp.unsuccessful_time_out
    assert isinstance(reset_dataset.actions.arm_action, mdp.RelativeJointPositionActionCfg)
    assert reset_dataset.actions.arm_action.scale == pytest.approx(0.015)
    assert reset_dataset.actions.gripper_action.use_incremental_target is False
    assert reset_dataset.actions.gripper_action.binary_threshold == pytest.approx(0.0)

    assert reverse.rewards.success.func is mdp.pour_success_bonus
    assert reverse.rewards.failure.func is mdp.terminal_failure
    assert reverse.rewards.failure.params == {}
    assert reverse.terminations.success.func is mdp.stable_pour_success
    assert "terminate" not in reverse.terminations.lost_grasp.params
    assert "terminate" not in reverse.terminations.spill.params
    assert reverse.terminations.time_out.func is mdp.unsuccessful_time_out
    assert reverse.actions.arm_action.scale == pytest.approx(0.08)
    assert reverse.actions.gripper_action.use_incremental_target is False
    assert reverse.actions.gripper_action.binary_threshold is None
    assert reverse.episode_length_s == pytest.approx(5.0)
    assert reverse.is_finite_horizon is True


def test_reset_dataset_ppo_is_calibrated_without_changing_traditional_curriculum():
    reset_dataset = FrankaPourResetDatasetPPORunnerCfg()
    reverse = FrankaPourPPORunnerCfg()

    assert reset_dataset.num_steps_per_env == 32
    assert reset_dataset.save_interval == 25
    assert reset_dataset.resume is False
    assert reset_dataset.actor.hidden_dims == [512, 256, 128, 64]
    assert reset_dataset.critic.hidden_dims == [512, 256, 128, 64]
    assert reset_dataset.actor.obs_normalization is False
    assert reset_dataset.critic.obs_normalization is False
    assert reset_dataset.actor.distribution_cfg.class_name == "HeteroscedasticGaussianDistribution"
    assert reset_dataset.actor.distribution_cfg.init_std == pytest.approx(0.25)
    assert reset_dataset.actor.distribution_cfg.std_range == pytest.approx((0.05, 0.75))
    assert reset_dataset.actor.distribution_cfg.std_type == "log"
    assert reset_dataset.algorithm.entropy_coef == pytest.approx(1.0e-3)
    assert reset_dataset.algorithm.gamma == pytest.approx(0.99 ** (1.0 / 3.0))
    assert reset_dataset.algorithm.lam == pytest.approx(0.95 ** (1.0 / 3.0))
    assert reset_dataset.algorithm.num_learning_epochs == 5
    assert reset_dataset.algorithm.num_mini_batches == 4
    assert reset_dataset.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert reset_dataset.algorithm.schedule == "fixed"

    assert reverse.num_steps_per_env == 32
    assert reverse.actor.hidden_dims == [256, 128, 64]
    assert reverse.critic.hidden_dims == [256, 128, 64]
    assert reverse.actor.obs_normalization is False
    assert reverse.critic.obs_normalization is False
    assert reverse.actor.distribution_cfg.init_std == pytest.approx(0.1)
    assert reverse.algorithm.entropy_coef == pytest.approx(1.0e-3)
    assert reverse.algorithm.gamma == pytest.approx(0.99)
    assert reverse.algorithm.lam == pytest.approx(0.95)


@pytest.mark.parametrize(
    "field, value, error_type, message",
    [
        ("reset_dataset_path", "", ValueError, "reset_dataset_path must be nonempty"),
        ("reset_dataset_path", None, TypeError, "reset_dataset_path must be a string"),
        ("reset_dataset_content_sha256", "a" * 63, ValueError, "lowercase SHA-256"),
        ("reset_dataset_content_sha256", "A" * 64, ValueError, "lowercase SHA-256"),
    ],
)
def test_reset_dataset_rejects_invalid_configuration(field, value, error_type, message):
    cfg = FrankaPourEnvCfg_RESET_DATASET()
    setattr(cfg, field, value)

    with pytest.raises(error_type, match=message):
        cfg.finalize()


@pytest.mark.parametrize(
    "field, value",
    [
        ("target_success_rate", 0.0),
        ("target_success_rate", 1.0),
        ("target_success_rate", float("nan")),
        ("temperature", 0.0),
        ("temperature", float("nan")),
        ("history_capacity", 0),
        ("history_capacity", 0.5),
        ("history_capacity", False),
        ("prior_strength", 0.0),
        ("prior_strength", float("inf")),
        ("initial_frontier_size", 0),
        ("initial_frontier_size", 0.5),
        ("probe_size", -1),
        ("probe_size", 0.5),
        ("probe_fraction", 1.0),
        ("probe_fraction", float("nan")),
        ("replay_fraction", 1.0),
        ("replay_fraction", float("nan")),
        ("frontier_evidence", 0.0),
        ("frontier_evidence", float("inf")),
    ],
)
def test_reset_dataset_rejects_invalid_sampler_configuration(field, value):
    cfg = FrankaPourEnvCfg_RESET_DATASET()
    setattr(cfg.reset_dataset_sampler, field, value)

    with pytest.raises(ValueError, match=field):
        cfg.finalize()


def test_reset_dataset_accepts_custom_path_and_content_hash():
    cfg = FrankaPourEnvCfg_RESET_DATASET()
    cfg.reset_dataset_path = "datasets/custom/reset_dataset.pt"
    cfg.reset_dataset_content_sha256 = "a" * 64

    resolved = cfg.finalize()

    assert resolved.reset_dataset_path == "datasets/custom/reset_dataset.pt"
    assert resolved.reset_dataset_content_sha256 == "a" * 64


def test_reset_dataset_top_grasp_filter_is_frozen_playback_only():
    cfg = FrankaPourEnvCfg_RESET_DATASET(
        reset_dataset_top_grasp_count=64,
        curriculum_freeze=True,
    ).finalize()
    assert cfg.reset_dataset_top_grasp_count == 64

    with pytest.raises(ValueError, match="positive integer"):
        FrankaPourEnvCfg_RESET_DATASET(
            reset_dataset_top_grasp_count=0,
            curriculum_freeze=True,
        ).finalize()
    with pytest.raises(ValueError, match="curriculum_freeze=True"):
        FrankaPourEnvCfg_RESET_DATASET(reset_dataset_top_grasp_count=64).finalize()


@pytest.mark.parametrize("threshold", [False, float("nan"), -1.0, 1.0])
def test_reset_dataset_rejects_invalid_binary_gripper_threshold(threshold):
    cfg = FrankaPourEnvCfg_RESET_DATASET()
    cfg.actions.gripper_action.binary_threshold = threshold

    with pytest.raises(ValueError, match="binary_threshold"):
        cfg.finalize()


def test_reset_dataset_rejects_binary_incremental_gripper_combination():
    cfg = FrankaPourEnvCfg_RESET_DATASET()
    cfg.actions.gripper_action.use_incremental_target = True

    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.finalize()


@pytest.mark.parametrize(
    "term_name, parameter, value",
    [
        ("reach", "std", 0.0),
        ("goal_distance", "std", float("nan")),
        ("joint_velocity", "max_velocity", 0.0),
    ],
)
def test_reset_dataset_rejects_invalid_general_reward(term_name, parameter, value):
    cfg = FrankaPourEnvCfg_RESET_DATASET()
    getattr(cfg.rewards, term_name).params[parameter] = value

    with pytest.raises(ValueError, match=term_name):
        cfg.finalize()


def test_finalize_resynchronizes_overridden_gripper_action_bound():
    cfg = FrankaPourEnvCfg()
    cfg.gripper_open_pos = 0.035
    cfg.success_max_tcp_distance = 0.02

    resolved = cfg.finalize()

    assert resolved.actions.gripper_action.open_position == pytest.approx(0.035)
    assert resolved.actions.gripper_action.default_position == pytest.approx(resolved.gripper_preload_pos)
    assert resolved.actions.gripper_action.scale == pytest.approx(0.035 - resolved.gripper_preload_pos)
    assert resolved.scene.robot.init_state.joint_pos["panda_finger_joint.*"] == pytest.approx(0.035)
    expected_contact_command = resolved.cup_grasp_box_half[1] - resolved.actions.gripper_action.contact_min_deflection
    assert resolved.rewards.task_progress.params["max_gripper_command"] == pytest.approx(expected_contact_command)
    assert resolved.rewards.delivered.params["max_gripper_command"] == pytest.approx(expected_contact_command)
    assert resolved.terminations.lost_grasp.params["max_gripper_command"] == pytest.approx(expected_contact_command)
    assert resolved.terminations.success.params["max_gripper_command"] == pytest.approx(expected_contact_command)


@pytest.mark.parametrize("command", [0.023, 0.028])
def test_finalize_rejects_gripper_success_commands_without_guaranteed_preload(command):
    cfg = FrankaPourEnvCfg()
    cfg.success_max_gripper_command = command

    with pytest.raises(ValueError, match="success_max_gripper_command"):
        cfg.finalize()


def test_backward_curriculum_rejects_invalid_stage_and_arm_overrides():
    invalid_stage = FrankaPourEnvCfg()
    invalid_stage.curriculum_start_stage = len(invalid_stage.curriculum_stage_names)
    with pytest.raises(ValueError, match="curriculum_start_stage"):
        invalid_stage.finalize()

    invalid_arm = FrankaPourEnvCfg()
    invalid_arm.curriculum_pour_arm_q = (10.0,) * 7
    with pytest.raises(ValueError, match="outside"):
        invalid_arm.finalize()

    invalid_replay = FrankaPourEnvCfg()
    invalid_replay.curriculum_previous_stage_replay_fraction = 1.0
    with pytest.raises(ValueError, match="curriculum_previous_stage_replay_fraction"):
        invalid_replay.finalize()

    invalid_entry_replay = FrankaPourEnvCfg()
    invalid_entry_replay.curriculum_frontier_entry_replay_fraction = 0.05
    with pytest.raises(ValueError, match="curriculum_frontier_entry_replay_fraction"):
        invalid_entry_replay.finalize()

    invalid_cohort_count = FrankaPourEnvCfg()
    invalid_cohort_count.curriculum_min_reset_cohorts_per_stage = -1.0
    with pytest.raises(ValueError, match="curriculum_min_reset_cohorts_per_stage"):
        invalid_cohort_count.finalize()

    invalid_transport = FrankaPourEnvCfg()
    invalid_transport.curriculum_transport_reset_fractions = (0.75, 0.25)
    with pytest.raises(ValueError, match="curriculum_transport_reset_fractions"):
        invalid_transport.finalize()

    invalid_grasp_approach = FrankaPourEnvCfg()
    invalid_grasp_approach.curriculum_grasp_approach_fractions = (0.75, 0.50, 0.70, 0.25, 0.125, 0.0)
    with pytest.raises(ValueError, match="curriculum_grasp_approach_fractions"):
        invalid_grasp_approach.finalize()

    unscreened_grasp_approach = FrankaPourEnvCfg()
    unscreened_grasp_approach.curriculum_grasp_approach_fractions = (0.70, 0.50, 0.375, 0.25, 0.125, 0.0)
    with pytest.raises(ValueError, match="eighth-segment"):
        unscreened_grasp_approach.finalize()

    invalid_randomization_threshold = FrankaPourEnvCfg()
    invalid_randomization_threshold.curriculum_randomization_promotion_threshold = 0.0
    with pytest.raises(ValueError, match="curriculum_randomization_promotion_threshold"):
        invalid_randomization_threshold.finalize()

    invalid_randomization_threshold.curriculum_randomization_promotion_threshold = 0.9
    with pytest.raises(ValueError, match="curriculum_randomization_promotion_threshold"):
        invalid_randomization_threshold.finalize()

    invalid_randomization_start = FrankaPourEnvCfg()
    invalid_randomization_start.curriculum_randomization_start_level = len(
        invalid_randomization_start.curriculum_randomization_extent_levels
    )
    with pytest.raises(ValueError, match="curriculum_randomization_start_level"):
        invalid_randomization_start.finalize()


@pytest.mark.parametrize("field,value", [("scale", 0.0), ("use_zero_offset", False), ("joint_names", ["panda_joint1"])])
def test_backward_curriculum_rejects_invalid_relative_arm_actions(field, value):
    cfg = FrankaPourEnvCfg()
    setattr(cfg.actions.arm_action, field, value)

    with pytest.raises(ValueError, match="Arm|Panda|relative"):
        cfg.finalize()


@pytest.mark.parametrize(
    "levels",
    [
        (),
        (-0.1, 1.0),
        (0.5, float("inf"), 1.0),
        (0.5, 0.5, 1.0),
        (0.75, 0.5, 1.0),
        (0.5, 1.0),
        (0.5, 0.9),
        (0.5, 1.1),
    ],
)
def test_randomized_curriculum_rejects_invalid_randomization_extent_levels(levels):
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomization_extent_levels = levels

    with pytest.raises(ValueError, match="curriculum_randomization_extent_levels"):
        cfg.finalize()


@pytest.mark.parametrize(
    "fractions",
    [
        (0.1, 0.35, 0.75, 1.0),
        (0.0, 0.35, 0.75),
        (0.0, 0.35, 1.1, 1.0),
        (0.0, 0.75, 0.35, 1.0),
        (0.0, 0.35, 0.75, 0.9),
    ],
)
def test_randomized_curriculum_rejects_invalid_independent_reset_fractions(fractions):
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_independent_arm_fraction_levels = fractions

    with pytest.raises(ValueError, match="curriculum_independent_arm_fraction_levels"):
        cfg.finalize()


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("target_tilt", 0.0),
        ("target_tilt", math.pi),
        ("pour_direction_xy", (0.0, 0.0)),
        ("pour_direction_xy", (0.0,)),
        ("alignment_radius", 0.0),
        ("active_through_stage", 8),
        ("discount_factor", 0.0),
    ],
)
def test_tilt_curriculum_rejects_invalid_configuration(parameter, value):
    cfg = FrankaPourEnvCfg()
    cfg.rewards.task_progress.params[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        cfg.finalize()


@pytest.mark.parametrize(
    "field,value",
    [
        ("curriculum_randomized_source_position_range", (-0.1, 0.1)),
        ("curriculum_randomized_source_radius_range", (0.68, 0.38)),
        ("curriculum_randomized_source_azimuth_range", 0.0),
        ("curriculum_randomized_source_xy_correlation", -0.01),
        ("curriculum_randomized_source_xy_correlation", 1.0),
        ("curriculum_randomized_source_xy_correlation", float("inf")),
        ("curriculum_randomized_carry_position_range", (0.31, 0.10)),
        ("curriculum_randomized_source_yaw_range", math.pi / 2.0),
        ("curriculum_randomized_target_position_range", (0.05, float("inf"))),
        ("curriculum_randomized_cup_clearance", -0.01),
        ("curriculum_randomized_reset_tcp_standoff", (0.0, 0.05)),
        ("curriculum_randomized_reset_tcp_jitter", (0.01, -0.01, 0.01)),
        ("curriculum_randomized_reset_tcp_rotation_angle_range", (-0.1, 0.2)),
        ("curriculum_randomized_reset_tcp_rotation_angle_range", (0.3, 0.2)),
        ("curriculum_randomized_reset_tcp_rotation_angle_range", (0.0, math.pi)),
        ("curriculum_randomized_reset_tcp_min_grasp_distance", 0.0),
        ("curriculum_randomized_reset_joint6_max", 0.0),
        ("curriculum_randomized_reset_joint6_max", float("inf")),
        ("curriculum_randomized_min_source_cell_fraction", 0.0),
        ("curriculum_randomized_min_source_cell_fraction", 1.01),
        ("curriculum_randomized_min_source_cell_fraction", float("inf")),
        ("curriculum_randomized_min_reset_variants_per_source", 0),
        ("curriculum_randomized_min_reset_variants_per_source", 12),
        ("curriculum_grasp_descent_overshoot", -0.001),
        ("curriculum_randomized_reset_ik_grid_size", 1),
        ("curriculum_randomized_reset_ik_samples_per_source", 1),
        ("curriculum_randomized_reset_ik_samples_per_source", 4),
        ("curriculum_randomized_reset_ik_iterations", 0),
    ],
)
def test_randomized_curriculum_rejects_invalid_configuration(field, value):
    cfg = FrankaPourEnvCfg()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        cfg.finalize()


def test_randomized_curriculum_rejects_ranges_without_collision_free_cup_placement():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_source_radius_range = None
    cfg.curriculum_randomized_source_position_range = (0.20, 0.50)

    with pytest.raises(ValueError, match="no collision-free target y-position"):
        cfg.finalize()


def test_randomized_curriculum_rejects_standoff_not_opposite_tool_axis():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_reset_tcp_standoff = (0.0, 0.0, 0.12)

    with pytest.raises(ValueError, match="antiparallel"):
        cfg.finalize()


def test_randomized_curriculum_rejects_vertical_tool_orientation():
    cfg = FrankaPourEnvCfg()
    cfg.cup_grasp_tcp_quat_c = (0.0, 0.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="parallel to the table"):
        cfg.finalize()


def test_randomized_curriculum_rejects_vertical_jaw_axis_with_horizontal_tool_axis():
    cfg = FrankaPourEnvCfg()
    # Tool +Z still points along cup +X, but a 90-degree tool roll makes Panda jaw +Y vertical.
    cfg.cup_grasp_tcp_quat_c = (0.5, 0.5, 0.5, 0.5)

    with pytest.raises(ValueError, match="jaw axis parallel to the table"):
        cfg.finalize()


def test_legacy_symmetric_tcp_jitter_rejects_missing_table_clearance():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_reset_tcp_offset_lower = None
    cfg.curriculum_randomized_reset_tcp_offset_upper = None
    cfg.curriculum_randomized_reset_tcp_jitter = (0.02, 0.04, 0.082)

    with pytest.raises(ValueError, match="above the table"):
        cfg.finalize()


def test_asymmetric_tcp_offset_rejects_below_pregrasp_start():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_reset_tcp_offset_lower = (-0.20, -0.10, -0.001)

    with pytest.raises(ValueError, match="must not place the initial TCP below"):
        cfg.finalize()


def test_randomized_curriculum_rejects_tcp_box_below_minimum_grasp_distance():
    cfg = FrankaPourEnvCfg()
    cfg.curriculum_randomized_reset_tcp_min_grasp_distance = 0.13

    with pytest.raises(ValueError, match="cannot guarantee curriculum_randomized_reset_tcp_min_grasp_distance"):
        cfg.finalize()


@pytest.mark.parametrize(
    "field,value",
    [
        ("success_dwell_time_s", 0.0),
        ("success_min_lift_height", 0.0),
        ("success_max_tcp_distance", float("inf")),
        ("success_max_gripper_width_error", 0.0),
        ("state_bound_joint_position_margin", -0.1),
        ("state_bound_max_joint_velocity", 0.0),
        ("state_bound_max_cup_linear_velocity", float("inf")),
        ("state_bound_max_cup_angular_velocity", 0.0),
        ("curriculum_randomized_pour_clearance", -0.001),
    ],
)
def test_success_and_state_bounds_reject_invalid_configuration(field, value):
    cfg = FrankaPourEnvCfg()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        cfg.finalize()


def test_finger_actuator_is_kept_and_teleop_disables_timeout():
    cfg = FrankaPourEnvCfg()
    hand = cfg.scene.robot.actuators["panda_hand"]
    assert hand.joint_names_expr == ["panda_finger_joint.*"]
    assert hand.stiffness is None
    assert hand.damping is None
    assert hand.armature is None
    assert FrankaPourEnvCfg_PLAY().scene.num_envs == 1
    teleop_cfg = FrankaPourEnvCfg_TELEOP()
    assert teleop_cfg.terminations.time_out is None
    assert teleop_cfg.actions.arm_action.clip == {
        joint_name: limits
        for joint_name, limits in zip(
            teleop_cfg.actions.arm_action.joint_names,
            pour_env_cfg.PANDA_ARM_JOINT_LIMITS,
            strict=True,
        )
    }
    assert isinstance(teleop_cfg.finalize().actions.arm_action, mdp.CurriculumJointPositionActionCfg)


def _media_capacity(cfg):
    return _media_entry(cfg).solver_cfg.max_active_cell_count


def _media_entry(cfg):
    return next(entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == "media")


@pytest.mark.parametrize("num_envs", [1, 4, 8, 64, 200, 1024, 2048])
def test_sparse_training_reserves_capturable_isolated_grid_capacity(num_envs):
    cfg = FrankaPourEnvCfg()
    cfg.scene.num_envs = num_envs
    resolved = cfg.finalize()

    assert _media_entry(resolved).solver_cfg.grid_type == "sparse"
    assert _media_capacity(resolved) == 512 * num_envs
    solver_cfg = _media_entry(resolved).solver_cfg
    assert solver_cfg.separate_worlds is True
    assert solver_cfg.max_lower_node_count == max(32, 16 * num_envs)
    assert solver_cfg.max_upper_node_count == max(32, (num_envs + 1) // 2)
    assert resolved.scene.env_spacing == pytest.approx(cfg.scene.env_spacing)
    assert resolved.sim.physics.use_cuda_graph is True


def test_play_uses_sparse_grid_capacity_and_honors_exact_override():
    play_cfg = FrankaPourEnvCfg_PLAY()

    assert _media_entry(play_cfg).solver_cfg.grid_type == "sparse"
    assert _resolve_mpm_cell_cap(play_cfg) == 512
    play_cfg.mpm_cell_cap_override = 23456
    assert _resolve_mpm_cell_cap(play_cfg) == 23456


def test_sparse_play_capacity_rejects_nonpositive_alignment():
    cfg = FrankaPourEnvCfg_PLAY()
    cfg.mpm_cell_capacity_alignment = 0

    with pytest.raises(ValueError, match="alignment must be positive"):
        _resolve_mpm_cell_cap(cfg)


def test_mpm_uses_captured_sparse_training_and_play_configs():
    cfg = FrankaPourEnvCfg()
    cfg.scene.num_envs = 200
    play_cfg = FrankaPourEnvCfg_PLAY().finalize()

    solver_cfg = _media_entry(cfg.finalize()).solver_cfg
    play_solver_cfg = _media_entry(play_cfg).solver_cfg

    # PIC27 bounds collider nodes by particle samples while Q1 keeps both solves compact.
    assert solver_cfg.velocity_basis == "Q1"
    assert solver_cfg.collider_basis == "pic27"
    assert solver_cfg.collider_velocity_mode == "forward"
    assert solver_cfg.project_outside_colliders is False
    assert solver_cfg.grid_type == "sparse"
    assert solver_cfg.max_active_cell_count == 200 * 512
    assert solver_cfg.separate_worlds is True
    assert play_solver_cfg.collider_basis == "pic27"
    assert play_solver_cfg.grid_type == "sparse"
    assert play_solver_cfg.grid_padding == 0
    assert play_solver_cfg.max_active_cell_count == 512
    assert play_solver_cfg.separate_worlds is True
    assert play_cfg.sim.physics.use_cuda_graph is True


def test_coarse_voxel_resolution_finalizes_without_manual_hierarchy_overrides():
    cfg = FrankaPourEnvCfg(voxel_size=0.03)
    cfg.scene.num_envs = 1

    assert cfg.voxel_size == pytest.approx(0.03)
    particles, _ = cup_cavity_lattice(cfg)
    assert particles.shape[0] > 0

    solver_cfg = _media_entry(cfg.finalize()).solver_cfg

    assert solver_cfg.voxel_size == pytest.approx(0.03)
    assert solver_cfg.grid_type == "sparse"
    assert solver_cfg.max_active_cell_count == 512


def test_particle_workspace_bounds_contain_every_curriculum_reset():
    cfg = FrankaPourEnvCfg()
    assert cfg.terminations.particle_out_of_bounds.func is mdp.particle_out_of_bounds
    assert cfg.particle_max_velocity == pytest.approx(10.0)
    lower = torch.tensor(cfg.particle_workspace_lower_bound)
    upper = torch.tensor(cfg.particle_workspace_upper_bound)
    local_particles = torch.from_numpy(cup_cavity_lattice(cfg)[0])

    source_range = torch.tensor((*cfg.curriculum_randomized_source_position_range, 0.0))
    source_center = torch.tensor(cfg.cup_reset_pos)
    for signed_range in (-source_range, source_range):
        particles = local_particles + source_center + signed_range
        assert bool(torch.all(particles >= lower))
        assert bool(torch.all(particles <= upper))

    invalid = FrankaPourEnvCfg()
    invalid.particle_workspace_lower_bound = (1.0, 0.0, 0.0)
    invalid.particle_workspace_upper_bound = (0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="particle_workspace"):
        invalid.finalize()

    randomized_outside = FrankaPourEnvCfg()
    randomized_outside.particle_workspace_upper_bound = (0.55, 1.0, 1.5)
    with pytest.raises(ValueError, match="randomized source media"):
        randomized_outside.finalize()

    invalid_velocity = FrankaPourEnvCfg()
    invalid_velocity.particle_max_velocity = 0.0
    with pytest.raises(ValueError, match="particle_max_velocity"):
        invalid_velocity.finalize()

    invalid_spill_fraction = FrankaPourEnvCfg()
    invalid_spill_fraction.max_spill_fraction = 1.0
    with pytest.raises(ValueError, match="max_spill_fraction"):
        invalid_spill_fraction.finalize()

    invalid_spill_height = FrankaPourEnvCfg()
    invalid_spill_height.spill_table_height = float("inf")
    with pytest.raises(ValueError, match="spill_table_height"):
        invalid_spill_height.finalize()

    invalid_count_margin = FrankaPourEnvCfg()
    invalid_count_margin.particle_count_margin = -0.001
    with pytest.raises(ValueError, match="particle_count_margin"):
        invalid_count_margin.finalize()


def test_capacity_resolver_reads_fixed_grid_type_from_media_entry_without_mutation():
    cfg = FrankaPourEnvCfg()
    media = _media_entry(cfg)
    media.solver_cfg.grid_type = "fixed"
    media.solver_cfg.max_active_cell_count = 120000
    arm = next(entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == "arm")
    arm.solver_cfg.max_active_cell_count = 777

    resolved_capacity = _resolve_mpm_cell_cap(cfg)

    assert resolved_capacity == 120000
    assert _media_capacity(cfg) == 120000
    assert arm.solver_cfg.max_active_cell_count == 777


def test_media_selector_includes_spill_floor_without_unrelated_shapes():
    cfg = FrankaPourEnvCfg().finalize()
    media = _media_entry(cfg)
    model = SimpleNamespace(
        body_label=[
            "/World/envs/env_0/TargetCup",
            "/World/envs/env_0/SpillFloor",
            "/World/envs/env_0/Robot/panda_hand",
        ],
        shape_count=4,
        shape_body=torch.tensor([0, 1, 2, -1], dtype=torch.int32),
        particle_count=3,
    )

    resolved = NewtonCoupler._resolve_entry(model, media, cfg.scene)

    assert resolved.bodies == [1]
    assert resolved.shapes == [1]
    assert resolved.particles == [0, 1, 2]


@pytest.mark.parametrize(
    "task_id, cfg_name, runner_name",
    [
        ("Isaac-Pour-Franka-v0", "FrankaPourEnvCfg", "FrankaPourPPORunnerCfg"),
        ("Isaac-Pour-Franka-Play-v0", "FrankaPourEnvCfg_PLAY", "FrankaPourPPORunnerCfg"),
        ("Isaac-Pour-Franka-Teleop-v0", "FrankaPourEnvCfg_TELEOP", None),
        (
            "Isaac-Pour-Franka-Reset-Dataset-v0",
            "FrankaPourEnvCfg_RESET_DATASET",
            "FrankaPourResetDatasetPPORunnerCfg",
        ),
        (
            "Isaac-Pour-Franka-Reset-Dataset-Eval-v0",
            "FrankaPourEnvCfg_RESET_DATASET_EVAL",
            "FrankaPourResetDatasetPPORunnerCfg",
        ),
        (
            "Isaac-Pour-Franka-Reset-Dataset-Play-v0",
            "FrankaPourEnvCfg_RESET_DATASET_PLAY",
            "FrankaPourResetDatasetPPORunnerCfg",
        ),
        (
            "Isaac-Pour-Franka-Reset-Mixture-v0",
            "FrankaPourEnvCfg_RESET_MIXTURE",
            "FrankaPourResetMixturePPORunnerCfg",
        ),
        (
            "Isaac-Pour-Franka-Reset-Mixture-Eval-v0",
            "FrankaPourEnvCfg_RESET_MIXTURE_EVAL",
            "FrankaPourResetMixturePPORunnerCfg",
        ),
        (
            "Isaac-Pour-Franka-Reset-Mixture-Play-v0",
            "FrankaPourEnvCfg_RESET_MIXTURE_PLAY",
            "FrankaPourResetMixturePPORunnerCfg",
        ),
    ],
)
def test_gym_registration_exactly_matches_task_entry_points(task_id, cfg_name, runner_name):
    spec = gym.spec(task_id)
    assert spec.entry_point == "isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv"
    assert spec.disable_env_checker is True
    expected_kwargs = {
        "env_cfg_entry_point": f"isaaclab_tasks.contrib.franka_pour.pour_env_cfg:{cfg_name}",
    }
    if runner_name is not None:
        expected_kwargs["rsl_rl_cfg_entry_point"] = (
            f"isaaclab_tasks.contrib.franka_pour.config.franka.agents.rsl_rl_ppo_cfg:{runner_name}"
        )
    assert spec.kwargs == expected_kwargs


def test_deprecated_reset_mixture_names_alias_reset_dataset_implementation():
    assert FrankaPourResetMixturePPORunnerCfg is FrankaPourResetDatasetPPORunnerCfg
    assert FrankaPourEnvCfg_RESET_MIXTURE is FrankaPourEnvCfg_RESET_DATASET
    assert FrankaPourEnvCfg_RESET_MIXTURE_EVAL is FrankaPourEnvCfg_RESET_DATASET_EVAL
    assert FrankaPourEnvCfg_RESET_MIXTURE_PLAY is FrankaPourEnvCfg_RESET_DATASET_PLAY
    assert mdp.PourResetMixture is mdp.PourResetDatasetCurriculum


def test_task_source_does_not_traverse_private_solver_state():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    private_manager_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and isinstance(node.value, ast.Name)
        and node.value.id in {"NewtonManager", "NewtonCoupler"}
    }
    assert private_manager_attrs == set()


def test_tcp_pose_uses_public_robot_pose_data_and_configured_offset():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    tcp_pose = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "tcp_pose_e")

    attributes = {node.attr for node in ast.walk(tcp_pose) if isinstance(node, ast.Attribute)}
    assert "get_term" not in attributes
    assert "_compute_frame_pose" not in attributes
    assert {"body_link_pose_w", "root_link_pose_w"} <= attributes
    assert {"subtract_frame_transforms", "combine_frame_transforms"} <= attributes

    setup = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_setup_after_physics"
    )
    setup_source = ast.unparse(setup)
    assert "cfg.tcp_body_name" in setup_source
    assert "cfg.tcp_offset_pos" in setup_source
    assert "cfg.tcp_offset_rot" in setup_source
    assert "cfg.actions.arm_action.body_name" not in setup_source
    assert "cfg.actions.arm_action.body_offset" not in setup_source


def test_media_refill_is_batched_on_device_without_host_readback():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    sample_media = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_sample_cup_media"
    )

    attributes = {node.attr for node in ast.walk(sample_media) if isinstance(node, ast.Attribute)}
    assert {"cpu", "numpy", "tolist"}.isdisjoint(attributes)
    assert "quat_apply" in attributes


def test_task_reset_uses_public_asset_writers_and_manager_lifecycle():
    source_path = Path(franka_pour.__file__).with_name("pour_env.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_reset_mpm_particle_state" not in function_names

    reset = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "reset_pour_scene"
    )
    call_names = {
        node.func.attr
        for node in ast.walk(reset)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {
        "write_joint_position_to_sim_index",
        "write_joint_velocity_to_sim_index",
        "write_root_pose_to_sim_index",
        "write_root_velocity_to_sim_index",
        "write_particle_pos_to_sim_index",
        "write_particle_velocity_to_sim_index",
    } <= call_names
    assert "reset_solver_state" in call_names
    assert {"eval_fk", "get_state_0", "get_state_1"}.isdisjoint(call_names)

    reset_attributes = {node.attr for node in ast.walk(reset) if isinstance(node, ast.Attribute)}
    assert "body_link_pose_w" in reset_attributes

    manager_forward_calls = [
        node
        for node in ast.walk(reset)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"forward", "forward_pending"}
    ]
    assert manager_forward_calls == []
