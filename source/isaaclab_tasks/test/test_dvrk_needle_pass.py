# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused contracts for the manager-based dVRK needle-pass task."""

from types import SimpleNamespace

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.managers import ObservationTermCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.needle_pass import assets
from isaaclab_tasks.manager_based.manipulation.needle_pass.config.dvrk.ik_abs_env_cfg import (
    DONOR_GRASP_CANDIDATE_INDEX,
    DONOR_GRASP_CLOSEDNESS,
    DONOR_GRASP_CONTACT_POINTS_N_M,
    DONOR_GRASP_JAW_POS,
    DONOR_GRASP_NATIVE_SEED_POS,
    DONOR_GRASP_NATIVE_SEED_ROT_WXYZ,
    DONOR_GRASP_OUTWARD_NORMALS_N,
    DONOR_GRASP_T_N_C_POS_M,
    DONOR_GRASP_T_N_C_ROT_WXYZ,
    DONOR_HELD_RESET_JAW_POS,
    DVRK_HANDOFF_PHASE_CFG,
    DVRK_JAW_CHANNEL_T_T_C_POS_M,
    DVRK_JAW_CHANNEL_T_T_C_ROT_WXYZ,
    DVRK_NEEDLE_PASS_JAW_ACTUATOR,
    ISAAC_GRASP_ASSET_SHA256,
    ISAAC_GRASP_CANDIDATES_SHA256,
    ISAAC_GRASP_CONFIG_SHA256,
    ISAAC_GRASP_GENERATOR_API,
    ISAAC_GRASP_GENERATOR_CANDIDATE_COUNT,
    ISAAC_GRASP_GENERATOR_CENTRE_COUNT,
    ISAAC_GRASP_GENERATOR_EXTENSION,
    ISAAC_GRASP_GENERATOR_EXTENSION_VERSION,
    ISAAC_GRASP_GENERATOR_ORIENTATIONS_PER_CENTRE,
    ISAAC_GRASP_GENERATOR_SEED,
    ISAAC_GRASP_GENERATOR_SIM_VERSION,
    ISAAC_GRASP_MANIFEST_SHA256,
    ISAAC_GRASP_WRAPPER_SHA256,
    LEFT_TOOL_HOME_POS_W,
    LEFT_TOOL_HOME_ROT_XYZW,
    NEEDLE_RESET_POS,
    NEEDLE_RESET_ROT_WXYZ,
    RECEIVER_ACQUISITION_NEEDLE_POS_W,
    RECEIVER_ACQUISITION_NEEDLE_ROT_WXYZ,
    RECEIVER_GRASP_CANDIDATE_INDEX,
    RECEIVER_GRASP_T_N_C_POS_M,
    RECEIVER_GRASP_T_N_C_ROT_WXYZ,
    RECEIVER_NEEDLE_TARGET_POS_T,
    RECEIVER_NEEDLE_TARGET_ROT_WXYZ,
    RECEIVER_TOOL_TARGET_POS_W,
    RECEIVER_TOOL_TARGET_ROT_WXYZ,
    RIGHT_TOOL_HOME_POS_W,
    DVRKNeedlePassEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.needle_pass.mdp.actions import (
    DonorReleaseGuardedPairedJawJointPositionAction,
    DonorReleaseGuardedPairedJawJointPositionActionCfg,
    donor_opening_requested,
    donor_release_is_allowed,
    world_pose_xyzw_to_root_pose_wxyz,
)
from isaaclab_tasks.manager_based.manipulation.needle_pass.mdp.events import reset_needle_pass_to_default
from isaaclab_tasks.manager_based.manipulation.needle_pass.mdp.grasp_solver import (
    EXACT_POINT_CONTACT_FORCE_RESIDUAL_TOLERANCE_N,
    EXACT_POINT_CONTACT_MOMENT_ARM_TOLERANCE_M,
    FORCE_CLOSURE_CONE_FACETS,
    assess_finite_contact_acceptance,
    friction_cone_generators,
    impedance_gains,
    prove_two_contact_force_closure,
    required_retention_load,
    two_contact_friction_wrench_generators,
)
from isaaclab_tasks.manager_based.manipulation.needle_pass.mdp.terminations import (
    JAW_BODY_REACTION_NORMALS_LOCAL,
    JAW_CONTACT_SENSOR_NAMES,
    HandoffMeasurements,
    HandoffPhase,
    HandoffPhaseCfg,
    HandoffPhaseMachine,
    jaw_needle_contact_measurements,
    update_handoff_phase,
)
from isaaclab_tasks.manager_based.manipulation.needle_pass.needle_pass_env_cfg import (
    DVRK_JAW_AUTHORED_DYNAMIC_FRICTION,
    HANDOFF_PHASE_CFG,
    REQUIRED_RETENTION_LOAD,
    RESOLVED_JAW_NEEDLE_DYNAMIC_FRICTION,
    RESOLVED_JAW_NEEDLE_STATIC_FRICTION,
    RETENTION_FRICTION_CONE_FACETS,
    UsdFileWithRigidMaterialCfg,
)
from isaaclab_tasks.utils import parse_env_cfg

TASK_ID = "Isaac-NeedlePass-dVRK-IK-Abs-v0"


def _term_names(group) -> list[str]:
    return [name for name, value in vars(group).items() if isinstance(value, ObservationTermCfg)]


def test_registration_action_order_and_observation_names():
    """Keep the task ID, 18D order, recorder names, and device key stable."""

    assert gym.spec(TASK_ID).entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    cfg = DVRKNeedlePassEnvCfg()
    assert isinstance(cfg.scene.needle.spawn, UsdFileWithRigidMaterialCfg)
    assert cfg.scene.needle.spawn.func.__name__ == "spawn_usd_with_rigid_material"
    assert list(vars(cfg.actions)) == [
        "left_arm_action",
        "left_jaw_action",
        "right_arm_action",
        "right_jaw_action",
    ]
    assert cfg.actions.left_arm_action.controller.command_type == "pose"
    assert cfg.actions.left_arm_action.controller.use_relative_mode is False
    assert len(cfg.actions.left_jaw_action.joint_names) == 2
    assert cfg.actions.right_arm_action.controller.command_type == "pose"
    assert cfg.actions.right_arm_action.controller.use_relative_mode is False
    assert len(cfg.actions.right_jaw_action.joint_names) == 2
    assert cfg.actions.left_jaw_action.scale == 1.0
    assert cfg.actions.left_jaw_action.offset == 0.0
    assert cfg.actions.left_jaw_action.use_default_offset is False
    assert cfg.actions.left_jaw_action.preserve_order is True
    assert isinstance(cfg.actions.left_jaw_action, DonorReleaseGuardedPairedJawJointPositionActionCfg)
    assert cfg.actions.left_jaw_action.phase_cfg == DVRK_HANDOFF_PHASE_CFG
    assert DVRK_HANDOFF_PHASE_CFG.opposed_normal_tolerance_rad == pytest.approx(np.deg2rad(25.0))
    assert cfg.actions.left_jaw_action.release_aperture_threshold_rad == pytest.approx(0.0)
    assert cfg.actions.left_jaw_action.hold_jaw_pos == DONOR_GRASP_JAW_POS
    assert cfg.actions.right_jaw_action.preserve_order is True
    assert list(cfg.teleop_devices.devices) == ["motion_controllers"]

    assert _term_names(cfg.observations.policy) == [
        "left_joint_pos",
        "left_joint_vel",
        "right_joint_pos",
        "right_joint_vel",
        "left_ee_pose_w",
        "right_ee_pose_w",
        "needle_pose_w",
        "needle_velocity_w",
        "jaw_needle_contact_force",
        "handoff_phase",
    ]
    assert _term_names(cfg.observations.subtask_terms) == [
        "donor_hold",
        "co_hold",
        "receiver_only_hold",
        "retained_lift",
    ]
    assert cfg.observations.policy.concatenate_terms is False
    assert cfg.observations.subtask_terms.concatenate_terms is False
    assert cfg.terminations.success.func.__name__ == "success"
    assert cfg.terminations.success.params["phase_cfg"] == DVRK_HANDOFF_PHASE_CFG
    assert cfg.terminations.success.params["phase_cfg"].receiver_relative_position_target_m == (
        RECEIVER_NEEDLE_TARGET_POS_T
    )
    assert cfg.terminations.success.params["phase_cfg"].receiver_relative_orientation_target_wxyz == (
        RECEIVER_NEEDLE_TARGET_ROT_WXYZ
    )
    assert cfg.terminations.success.params["phase_cfg"].receiver_relative_position_limit_m == 0.003
    assert cfg.terminations.success.params["phase_cfg"].receiver_relative_orientation_limit_rad == pytest.approx(
        np.deg2rad(15.0)
    )


def test_config_construction_does_not_contact_asset_host(monkeypatch):
    """Config discovery and registry inspection must remain offline-safe."""

    assets.verify_remote_asset_sha256.cache_clear()
    monkeypatch.setattr(
        assets,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("config attempted a network request")),
    )

    cfg = DVRKNeedlePassEnvCfg()

    assert cfg.scene.needle.spawn.usd_path == assets.NEEDLE_ASSET.url


def test_cuda_override_propagates_to_dvrk_teleoperation():
    """Keep the stock recorder's teleop action on the required CUDA device."""

    device = "cuda:0"
    cfg = parse_env_cfg(TASK_ID, device=device, num_envs=1)
    device_cfg = cfg.teleop_devices.devices["motion_controllers"]

    assert cfg.sim.device == device
    assert device_cfg.sim_device == device
    assert len(device_cfg.retargeters) == 1
    assert device_cfg.retargeters[0].sim_device == device


def test_dvrk_task_rejects_cpu_override():
    with pytest.raises(ValueError, match="requires a CUDA simulation device"):
        parse_env_cfg(TASK_ID, device="cpu", num_envs=1)


def test_asset_pins_and_explicit_needle_physics():
    assert assets.I4H_CATALOGUE_LICENCE == "Apache-2.0"
    assert assets.I4H_CATALOGUE_LICENCE_URL.endswith("/blob/v0.6.0/LICENSE")
    assert assets.NEEDLE_ASSET.key.endswith("needle_sdf.usd")
    assert assets.NEEDLE_ASSET.sha256 == "2b317a61f93631a7192e7ed2839ef20f7a75c05aa5f84a3905696134a64f36d7"
    assert assets.SUTURE_PAD_ASSET.sha256 == "1c6e4624097fbf8ffc49131539e9eec72d96c5cf68916fbe04eefff1e9522a51"
    assert assets.NEEDLE_SCALE == (0.4, 0.4, 0.4)
    np.testing.assert_allclose(
        assets.NEEDLE_BODY_LOCAL_AABB_MIN_M,
        np.asarray(assets.NEEDLE_SOURCE_AABB_MIN_M) * np.asarray(assets.NEEDLE_SCALE),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        assets.NEEDLE_BODY_LOCAL_AABB_MAX_M,
        np.asarray(assets.NEEDLE_SOURCE_AABB_MAX_M) * np.asarray(assets.NEEDLE_SCALE),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        assets.NEEDLE_BODY_LOCAL_EXTENT_M,
        (0.020065199650707657, 0.03957919925451279, 0.001651600003242493),
        atol=1.0e-15,
        rtol=0.0,
    )
    assert np.isclose(assets.NEEDLE_SOURCE_VOLUME_M3, 2.085934204311373e-6)
    assert assets.NEEDLE_REFERENCE_DENSITY_KG_M3 == 8000.0
    assert np.isclose(assets.NEEDLE_MASS_KG, 0.0010679983126074231)
    assert np.isclose(
        assets.NEEDLE_MASS_KG,
        assets.NEEDLE_SOURCE_VOLUME_M3 * np.prod(assets.NEEDLE_SCALE) * assets.NEEDLE_REFERENCE_DENSITY_KG_M3,
    )
    np.testing.assert_allclose(
        assets.NEEDLE_CENTRE_OF_MASS_BODY_LOCAL_M,
        np.asarray(assets.NEEDLE_SOURCE_CENTRE_OF_MASS_M) * np.asarray(assets.NEEDLE_SCALE),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        assets.NEEDLE_CENTRE_OF_MASS_BODY_LOCAL_M,
        (0.0064017449816068, 0.0004139220109208, 0.0003823855658992),
        atol=1.0e-15,
        rtol=0.0,
    )
    assert 0.0 < assets.NEEDLE_DYNAMIC_FRICTION <= assets.NEEDLE_STATIC_FRICTION

    cfg = DVRKNeedlePassEnvCfg()
    assert cfg.scene.needle.spawn.rigid_props.kinematic_enabled is False
    assert cfg.scene.needle.spawn.rigid_props.disable_gravity is False
    assert cfg.scene.needle.spawn.collision_props.collision_enabled is True
    assert cfg.scene.needle.spawn.mass_props.mass == assets.NEEDLE_MASS_KG
    assert cfg.scene.needle.spawn.physics_material.static_friction == assets.NEEDLE_STATIC_FRICTION
    assert cfg.scene.needle.spawn.physics_material.dynamic_friction == assets.NEEDLE_DYNAMIC_FRICTION
    assert cfg.scene.suture_pad.init_state.pos[:2] == (0.45, 0.45)
    for name in JAW_CONTACT_SENSOR_NAMES:
        sensor_cfg = getattr(cfg.scene, name)
        assert sensor_cfg.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Needle"]
        assert sensor_cfg.track_pose is True
    assert cfg.sim.dt == pytest.approx(1.0 / 240.0)
    assert cfg.decimation == 1
    assert cfg.sim.physx.solver_type == 1
    assert cfg.scene.left_psm.spawn.articulation_props.solver_velocity_iteration_count == 4
    assert DVRK_JAW_AUTHORED_DYNAMIC_FRICTION == 10.0
    assert assets.NEEDLE_FRICTION_COMBINE_MODE == "min"
    assert np.isclose(RESOLVED_JAW_NEEDLE_DYNAMIC_FRICTION, min(10.0, assets.NEEDLE_DYNAMIC_FRICTION))
    assert np.isclose(RESOLVED_JAW_NEEDLE_STATIC_FRICTION, min(1.0, assets.NEEDLE_STATIC_FRICTION))
    assert REQUIRED_RETENTION_LOAD.friction_coefficient == RESOLVED_JAW_NEEDLE_STATIC_FRICTION
    assert HANDOFF_PHASE_CFG.engage_force_n == REQUIRED_RETENTION_LOAD.normal_force_per_jaw_n
    assert HANDOFF_PHASE_CFG.engage_force_n > 0.011


def _matrix_from_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_world_targets_use_each_live_psm_root():
    """Check two different rotated roots against independent homogeneous matrices."""

    root_pos = np.array(((0.2, -0.1, 0.3), (-0.4, 0.2, 0.1)), dtype=np.float32)
    root_quat_wxyz = np.array(
        ((1.0, 0.0, 0.0, 0.0), (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))),
        dtype=np.float32,
    )
    target_quat_wxyz = np.array(
        ((np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)),
        dtype=np.float32,
    )
    target_pos = np.array(((0.3, 0.1, 0.5), (-0.1, 0.4, 0.2)), dtype=np.float32)
    pose_xyzw = np.concatenate((target_pos, target_quat_wxyz[:, (1, 2, 3, 0)]), axis=1)
    actual = world_pose_xyzw_to_root_pose_wxyz(
        torch.tensor(pose_xyzw),
        torch.tensor(root_pos),
        torch.tensor(root_quat_wxyz),
    ).numpy()

    for index in range(2):
        root_transform = np.eye(4)
        root_transform[:3, :3] = _matrix_from_quat_wxyz(root_quat_wxyz[index])
        root_transform[:3, 3] = root_pos[index]
        target_transform = np.eye(4)
        target_transform[:3, :3] = _matrix_from_quat_wxyz(target_quat_wxyz[index])
        target_transform[:3, 3] = target_pos[index]
        expected = np.linalg.inv(root_transform) @ target_transform
        np.testing.assert_allclose(actual[index, :3], expected[:3, 3], atol=1.0e-6)
        np.testing.assert_allclose(
            _matrix_from_quat_wxyz(actual[index, 3:7]),
            expected[:3, :3],
            atol=1.0e-6,
        )
    assert not np.allclose(actual[0], actual[1])


def test_grasp_solver_derives_loads_and_impedance_gains():
    load = required_retention_load(
        mass_kg=assets.NEEDLE_MASS_KG,
        gravity_m_s2=9.81,
        maximum_commanded_acceleration_m_s2=0.5,
        friction_coefficient=RESOLVED_JAW_NEEDLE_STATIC_FRICTION,
        safety_factor=2.0,
    )
    assert load.normal_force_per_jaw_n > 0.0
    cone = friction_cone_generators(
        (1.0, 0.0, 0.0),
        RESOLVED_JAW_NEEDLE_STATIC_FRICTION,
        RETENTION_FRICTION_CONE_FACETS,
    )
    assert cone.shape == (RETENTION_FRICTION_CONE_FACETS, 3)
    np.testing.assert_allclose(cone[:, 0], 1.0)
    np.testing.assert_allclose(
        np.linalg.norm(cone[:, 1:], axis=1),
        RESOLVED_JAW_NEEDLE_STATIC_FRICTION,
    )
    stiffness, damping = impedance_gains(3.0e-7, 60.0, 1.0)
    assert stiffness == pytest.approx(3.0e-7 * 60.0**2)
    assert damping == pytest.approx(2.0 * 3.0e-7 * 60.0)


def test_two_contact_force_closure_proves_gravity_wrench_deterministically():
    contact_points_m = ((-0.01, 0.0, 0.0), (0.01, 0.0, 0.0))
    inward_normals = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    static_friction_coefficient = 0.5
    required_wrench = (0.0, 0.0, 2.0, 0.0, 0.0, 0.0)

    generators = two_contact_friction_wrench_generators(
        contact_points_m,
        inward_normals,
        static_friction_coefficient,
    )
    assert FORCE_CLOSURE_CONE_FACETS == RETENTION_FRICTION_CONE_FACETS == 8
    assert generators.shape == (2 * FORCE_CLOSURE_CONE_FACETS, 6)

    proof = prove_two_contact_force_closure(
        contact_points_m,
        inward_normals,
        static_friction_coefficient,
        required_wrench,
    )
    repeated_proof = prove_two_contact_force_closure(
        contact_points_m,
        inward_normals,
        static_friction_coefficient,
        required_wrench,
    )
    assert proof.feasible is True
    assert proof.exact_point_contact_feasible is True
    assert proof.active_generator_indices == (0, 12)
    assert proof.coefficients == repeated_proof.coefficients
    assert proof.force_residual_norm_n == repeated_proof.force_residual_norm_n
    assert proof.torque_residual_norm_n_m == repeated_proof.torque_residual_norm_n_m
    assert proof.equivalent_moment_arm_residual_m == repeated_proof.equivalent_moment_arm_residual_m
    assert all(coefficient >= 0.0 for coefficient in proof.coefficients)
    assert proof.coefficients[0] == pytest.approx(2.0)
    assert proof.coefficients[12] == pytest.approx(2.0)
    np.testing.assert_allclose(proof.achieved_wrench, required_wrench, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(proof.coefficients) @ generators, required_wrench, atol=1.0e-12)
    assert proof.required_force_norm_n == pytest.approx(2.0)
    assert proof.force_residual_tolerance_n == EXACT_POINT_CONTACT_FORCE_RESIDUAL_TOLERANCE_N
    assert proof.moment_arm_residual_tolerance_m == EXACT_POINT_CONTACT_MOMENT_ARM_TOLERANCE_M
    assert proof.force_residual_norm_n <= 1.0e-12
    assert proof.torque_residual_norm_n_m <= 1.0e-12
    assert proof.equivalent_moment_arm_residual_m <= 1.0e-12


def test_two_contact_force_closure_rejects_unresisted_contact_axis_torque():
    proof = prove_two_contact_force_closure(
        ((-0.01, 0.0, 0.0), (0.01, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        0.5,
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    )
    assert proof.feasible is False
    assert proof.exact_point_contact_feasible is False
    assert proof.active_generator_indices == ()
    np.testing.assert_array_equal(proof.coefficients, np.zeros(2 * FORCE_CLOSURE_CONE_FACETS))
    np.testing.assert_allclose(proof.residual_wrench, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    assert proof.required_force_norm_n == 0.0
    assert proof.force_residual_norm_n == 0.0
    assert proof.torque_residual_norm_n_m == pytest.approx(1.0)
    assert proof.equivalent_moment_arm_residual_m == np.inf


def test_finite_contact_acceptance_is_distinct_from_exact_point_contact_proof():
    """A documented soft-contact allowance must not relabel exact closure."""

    proof = prove_two_contact_force_closure(
        ((-0.01, 0.0, 0.0), (0.01, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        0.5,
        (0.0, 0.0, 2.0, 1.0e-5, 0.0, 0.0),
    )
    assert proof.exact_point_contact_feasible is False
    assert proof.force_residual_norm_n <= 1.0e-12
    assert proof.torque_residual_norm_n_m == pytest.approx(1.0e-5)
    assert proof.equivalent_moment_arm_residual_m == pytest.approx(5.0e-6)

    accepted = assess_finite_contact_acceptance(
        proof,
        force_residual_tolerance_n=1.0e-9,
        moment_arm_residual_tolerance_m=10.0e-6,
    )
    assert accepted.accepted is True
    assert accepted.force_within_tolerance is True
    assert accepted.moment_arm_within_tolerance is True
    assert proof.exact_point_contact_feasible is False

    rejected = assess_finite_contact_acceptance(
        proof,
        force_residual_tolerance_n=1.0e-9,
        moment_arm_residual_tolerance_m=1.0e-6,
    )
    assert rejected.accepted is False
    assert rejected.force_within_tolerance is True
    assert rejected.moment_arm_within_tolerance is False


def test_two_contact_force_closure_validates_shapes_and_finite_inputs():
    points = ((-0.01, 0.0, 0.0), (0.01, 0.0, 0.0))
    normals = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    wrench = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"contact_points_m must be a finite \(2, 3\) array"):
        two_contact_friction_wrench_generators(points[:1], normals, 0.5)
    with pytest.raises(ValueError, match=r"inward_normals must be a finite \(2, 3\) array"):
        two_contact_friction_wrench_generators(points, ((np.nan, 0.0, 0.0), normals[1]), 0.5)
    with pytest.raises(ValueError, match="static_friction_coefficient must be finite and positive"):
        two_contact_friction_wrench_generators(points, normals, np.inf)
    with pytest.raises(ValueError, match="required_wrench must be a finite six-vector"):
        prove_two_contact_force_closure(points, normals, 0.5, wrench[:5])
    with pytest.raises(ValueError, match="required_wrench must be a finite six-vector"):
        prove_two_contact_force_closure(points, normals, 0.5, (*wrench[:5], np.nan))
    proof = prove_two_contact_force_closure(points, normals, 0.5, wrench)
    with pytest.raises(ValueError, match="force_residual_tolerance_n must be finite and positive"):
        assess_finite_contact_acceptance(
            proof,
            force_residual_tolerance_n=np.nan,
            moment_arm_residual_tolerance_m=10.0e-6,
        )
    with pytest.raises(ValueError, match="moment_arm_residual_tolerance_m must be finite and positive"):
        assess_finite_contact_acceptance(
            proof,
            force_residual_tolerance_n=1.0e-9,
            moment_arm_residual_tolerance_m=0.0,
        )


def _transform_from_pose(position, quaternion_wxyz):
    transform = np.eye(4)
    transform[:3, :3] = _matrix_from_quat_wxyz(np.asarray(quaternion_wxyz))
    transform[:3, 3] = position
    return transform


def test_native_isaac_grasp_generator_provenance_and_held_reset():
    """Lock the native generator run and the physically held closed reset."""

    assert ISAAC_GRASP_GENERATOR_EXTENSION == "isaacsim.replicator.grasping"
    assert ISAAC_GRASP_GENERATOR_EXTENSION_VERSION == "1.0.9"
    assert ISAAC_GRASP_GENERATOR_API.endswith("GraspingManager.generate_grasp_poses")
    assert ISAAC_GRASP_GENERATOR_SIM_VERSION == "5.1.0"
    assert ISAAC_GRASP_GENERATOR_SEED == 12
    assert ISAAC_GRASP_GENERATOR_CANDIDATE_COUNT == 8192
    assert ISAAC_GRASP_GENERATOR_ORIENTATIONS_PER_CENTRE == 32
    assert ISAAC_GRASP_GENERATOR_CENTRE_COUNT == 256
    assert ISAAC_GRASP_GENERATOR_CANDIDATE_COUNT == (
        ISAAC_GRASP_GENERATOR_CENTRE_COUNT * ISAAC_GRASP_GENERATOR_ORIENTATIONS_PER_CENTRE
    )
    assert assets.NEEDLE_ASSET.sha256 == ISAAC_GRASP_ASSET_SHA256
    assert ISAAC_GRASP_WRAPPER_SHA256 == "01bc820d1777a1655a5c42b3ebac997c6281335a12d12f7636c3e25721f3a2d5"
    assert ISAAC_GRASP_CONFIG_SHA256 == "b308ec31bf9bf425c686007e0dc0ad72f09ae7e1f67e1015ac53dc92a017e798"
    assert ISAAC_GRASP_CANDIDATES_SHA256 == "7c601982d72759ca901fad9b59fa1df80a092221d1cb91eda88938b2b83bc374"
    assert ISAAC_GRASP_MANIFEST_SHA256 == "13c72a5fb58db7c211619b72dcbdf27890a25a35a5aa8e3185ab4ae3139970ee"

    assert DONOR_GRASP_CANDIDATE_INDEX == 2321
    assert divmod(DONOR_GRASP_CANDIDATE_INDEX, ISAAC_GRASP_GENERATOR_ORIENTATIONS_PER_CENTRE) == (72, 17)
    assert RECEIVER_GRASP_CANDIDATE_INDEX == 51
    assert divmod(RECEIVER_GRASP_CANDIDATE_INDEX, ISAAC_GRASP_GENERATOR_ORIENTATIONS_PER_CENTRE) == (1, 19)
    for quaternion in (DONOR_GRASP_T_N_C_ROT_WXYZ, RECEIVER_GRASP_T_N_C_ROT_WXYZ):
        assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1.0e-15)

    np.testing.assert_allclose(DONOR_GRASP_JAW_POS, (0.0, 0.0), atol=0.0, rtol=0.0)
    assert DONOR_GRASP_CLOSEDNESS == 1.0

    cfg = DVRKNeedlePassEnvCfg()
    for joint_name, position in zip(cfg.actions.left_jaw_action.joint_names, DONOR_HELD_RESET_JAW_POS, strict=True):
        assert cfg.scene.left_psm.init_state.joint_pos[joint_name] == position
    for joint_name, position in zip(cfg.actions.right_jaw_action.joint_names, (-np.pi / 6.0, np.pi / 6.0), strict=True):
        assert cfg.scene.right_psm.init_state.joint_pos[joint_name] == pytest.approx(position)
    retargeter_cfg = cfg.teleop_devices.devices["motion_controllers"].retargeters[0]
    assert retargeter_cfg.left.initial_closedness == DONOR_GRASP_CLOSEDNESS
    assert retargeter_cfg.right.initial_closedness == 0.0
    left_retargeter_reset = np.asarray(retargeter_cfg.left.jaw_open) + DONOR_GRASP_CLOSEDNESS * (
        np.asarray(retargeter_cfg.left.jaw_closed) - np.asarray(retargeter_cfg.left.jaw_open)
    )
    np.testing.assert_allclose(left_retargeter_reset, DONOR_GRASP_JAW_POS, atol=1.0e-15)
    assert cfg.scene.left_psm.actuators["jaws"] == DVRK_NEEDLE_PASS_JAW_ACTUATOR


def test_native_grasp_transforms_reconstruct_seed_and_receiver_controller_target():
    """Rebuild native seeds while keeping physical holding equilibria explicit."""

    left_home_wxyz = (
        LEFT_TOOL_HOME_ROT_XYZW[3],
        LEFT_TOOL_HOME_ROT_XYZW[0],
        LEFT_TOOL_HOME_ROT_XYZW[1],
        LEFT_TOOL_HOME_ROT_XYZW[2],
    )
    transform_w_donor_tool = _transform_from_pose(LEFT_TOOL_HOME_POS_W, left_home_wxyz)
    transform_tool_channel = _transform_from_pose(
        DVRK_JAW_CHANNEL_T_T_C_POS_M,
        DVRK_JAW_CHANNEL_T_T_C_ROT_WXYZ,
    )
    transform_needle_donor_channel = _transform_from_pose(
        DONOR_GRASP_T_N_C_POS_M,
        DONOR_GRASP_T_N_C_ROT_WXYZ,
    )
    transform_needle_receiver_channel = _transform_from_pose(
        RECEIVER_GRASP_T_N_C_POS_M,
        RECEIVER_GRASP_T_N_C_ROT_WXYZ,
    )

    expected_needle_w = transform_w_donor_tool @ transform_tool_channel @ np.linalg.inv(transform_needle_donor_channel)
    configured_native_seed_w = _transform_from_pose(DONOR_GRASP_NATIVE_SEED_POS, DONOR_GRASP_NATIVE_SEED_ROT_WXYZ)
    np.testing.assert_allclose(configured_native_seed_w, expected_needle_w, atol=1.0e-9)
    configured_needle_w = _transform_from_pose(NEEDLE_RESET_POS, NEEDLE_RESET_ROT_WXYZ)
    assert not np.allclose(configured_needle_w, expected_needle_w, atol=1.0e-5)

    native_receiver_needle = transform_tool_channel @ np.linalg.inv(transform_needle_receiver_channel)
    acceptance_receiver_needle = _transform_from_pose(
        RECEIVER_NEEDLE_TARGET_POS_T,
        RECEIVER_NEEDLE_TARGET_ROT_WXYZ,
    )
    assert not np.allclose(acceptance_receiver_needle, native_receiver_needle, atol=1.0e-5)

    acquisition_needle_w = _transform_from_pose(
        RECEIVER_ACQUISITION_NEEDLE_POS_W,
        RECEIVER_ACQUISITION_NEEDLE_ROT_WXYZ,
    )
    assert not np.allclose(acquisition_needle_w, configured_needle_w, atol=1.0e-5)
    assert np.linalg.norm(acquisition_needle_w[:3, 3] - configured_needle_w[:3, 3]) < 1.0e-3

    expected_receiver_tool_w = acquisition_needle_w @ np.linalg.inv(native_receiver_needle)
    configured_receiver_tool_w = _transform_from_pose(
        RECEIVER_TOOL_TARGET_POS_W,
        RECEIVER_TOOL_TARGET_ROT_WXYZ,
    )
    np.testing.assert_allclose(configured_receiver_tool_w, expected_receiver_tool_w, atol=1.0e-9)
    np.testing.assert_allclose(configured_receiver_tool_w @ native_receiver_needle, acquisition_needle_w, atol=1.0e-9)


def test_native_donor_grasp_does_not_claim_unproven_point_contact_closure():
    """Keep the analytical two-point model separate from physical retention."""

    contact_points_from_com = np.asarray(DONOR_GRASP_CONTACT_POINTS_N_M) - np.asarray(
        assets.NEEDLE_CENTRE_OF_MASS_BODY_LOCAL_M
    )
    inward_normals = -np.asarray(DONOR_GRASP_OUTWARD_NORMALS_N)
    required_force_w = np.array(
        (0.0, 0.0, REQUIRED_RETENTION_LOAD.safety_factor * REQUIRED_RETENTION_LOAD.external_force_n)
    )
    required_force_n = _matrix_from_quat_wxyz(np.asarray(NEEDLE_RESET_ROT_WXYZ)).T @ required_force_w
    proof = prove_two_contact_force_closure(
        contact_points_from_com,
        inward_normals,
        RESOLVED_JAW_NEEDLE_STATIC_FRICTION,
        np.concatenate((required_force_n, np.zeros(3))),
    )
    assert proof.exact_point_contact_feasible is False
    # The candidate is selected by its observed full collision-patch hold, not
    # by a fictitious two-point proof.  Its non-zero point-model moment error
    # must remain visible so later changes cannot turn it into a false claim.
    assert proof.force_residual_norm_n > EXACT_POINT_CONTACT_FORCE_RESIDUAL_TOLERANCE_N
    assert proof.equivalent_moment_arm_residual_m > EXACT_POINT_CONTACT_MOMENT_ARM_TOLERANCE_M

    finite_patch = assess_finite_contact_acceptance(
        proof,
        force_residual_tolerance_n=3.0e-8,
        moment_arm_residual_tolerance_m=10.0e-6,
    )
    assert finite_patch.accepted is False
    assert proof.exact_point_contact_feasible is False


def _measurements(num_envs: int, loads: torch.Tensor, needle_z: float = 0.0) -> HandoffMeasurements:
    normals = torch.tensor(JAW_BODY_REACTION_NORMALS_LOCAL, dtype=torch.float32).repeat(num_envs, 1, 1)
    needle_pose = torch.zeros((num_envs, 7), dtype=torch.float32)
    needle_pose[:, 2] = needle_z
    needle_pose[:, 3] = 1.0
    receiver_pose = needle_pose.clone()
    return HandoffMeasurements(
        normal_forces_n=loads,
        reaction_normals_w=normals,
        needle_pose_w=needle_pose,
        needle_velocity_w=torch.zeros((num_envs, 6), dtype=torch.float32),
        receiver_pose_w=receiver_pose,
    )


def test_reset_observation_does_not_advance_before_first_action():
    """A held reset remains INITIAL until a fresh measured-contact sample."""

    cfg = HandoffPhaseCfg(
        donor_dwell_s=0.02,
        co_hold_dwell_s=0.02,
        receiver_only_dwell_s=0.02,
        retained_lift_dwell_s=0.02,
    )
    machine = HandoffPhaseMachine(1, "cpu", 0.01, cfg)
    machine.reset(torch.tensor([0]), torch.tensor([0.0]), step_token=7)
    donor_loads = torch.tensor(((1.0, 1.0, 0.0, 0.0),))
    machine.advance(_measurements(1, donor_loads), step_token=7)
    assert machine.phase.item() == HandoffPhase.INITIAL
    assert machine._donor_counter.item() == 0
    machine.advance(_measurements(1, donor_loads), step_token=8)
    assert machine._donor_counter.item() == 1


@pytest.mark.parametrize(
    ("loads", "normals"),
    (
        (((float("inf"), 1.0),), (((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),)),
        (((1.0, 1.0),), (((float("nan"), 0.0, 0.0), (-1.0, 0.0, 0.0)),)),
    ),
)
def test_bilateral_contact_rejects_nonfinite_measurements(loads, normals):
    """Non-finite contact data must never qualify a physical grasp."""

    machine = HandoffPhaseMachine(1, "cpu", 0.01, HandoffPhaseCfg())
    result = machine._bilateral_contact(
        torch.tensor(loads, dtype=torch.float32),
        torch.tensor(normals, dtype=torch.float32),
        torch.zeros(1, dtype=torch.bool),
    )
    assert result.tolist() == [False]


def test_handoff_update_reads_post_physics_buffers_once_per_step(monkeypatch):
    """Manager terms sharing one phase machine must also share one sensor sample."""

    from isaaclab_tasks.manager_based.manipulation.needle_pass.mdp import terminations

    calls = {"contacts": 0, "advance": 0}

    class _Machine:
        def advance(self, measurements, step_token):
            calls["advance"] += 1
            assert measurements.normal_forces_n.shape == (1, 4)
            assert step_token in (12, 13)

    machine = _Machine()

    def _contacts(_env):
        calls["contacts"] += 1
        return torch.zeros((1, 4)), torch.zeros((1, 4, 3)), torch.zeros((1, 4, 3))

    needle = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros((1, 3)),
            root_quat_w=torch.tensor(((1.0, 0.0, 0.0, 0.0),)),
            root_lin_vel_w=torch.zeros((1, 3)),
            root_ang_vel_w=torch.zeros((1, 3)),
        )
    )
    env = SimpleNamespace(common_step_counter=12, scene={"needle": needle, "right_psm": object()})
    monkeypatch.setattr(terminations, "get_handoff_phase_machine", lambda *_: machine)
    monkeypatch.setattr(terminations, "jaw_needle_contact_measurements", _contacts)
    monkeypatch.setattr(
        terminations,
        "_asset_pose_w",
        lambda *_: torch.tensor(((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),)),
    )

    assert update_handoff_phase(env, HandoffPhaseCfg()) is machine
    assert update_handoff_phase(env, HandoffPhaseCfg()) is machine
    assert calls == {"contacts": 1, "advance": 1}

    env.common_step_counter = 13
    assert update_handoff_phase(env, HandoffPhaseCfg()) is machine
    assert calls == {"contacts": 2, "advance": 2}


def test_donor_release_requires_a_current_receiver_grasp():
    """A completed co-hold dwell must not leave a stale release permission."""

    phase = torch.tensor((int(HandoffPhase.CO_HOLD), int(HandoffPhase.RECEIVER_ONLY_HOLD), int(HandoffPhase.CO_HOLD)))
    receiver_grasp = torch.tensor((True, False, False))
    allowed = donor_release_is_allowed(phase, receiver_grasp, int(HandoffPhase.CO_HOLD))
    assert allowed.tolist() == [True, False, False]

    with pytest.raises(ValueError, match="identical batch shapes"):
        donor_release_is_allowed(phase, receiver_grasp[:2], int(HandoffPhase.CO_HOLD))


def test_donor_release_blocks_any_outward_jaw_command():
    """A unilateral outward target must not bypass the donor-release interlock."""

    held = torch.tensor(((-0.20, 0.01),))
    more_closed = torch.tensor(((-0.10, 0.0),))
    open_one_jaw = torch.tensor(((-0.40, 0.01),))
    paired_open = torch.tensor(((-0.40, 0.30),))
    assert donor_opening_requested(held, held, 0.01).tolist() == [False]
    assert donor_opening_requested(more_closed, held, 0.01).tolist() == [False]
    assert donor_opening_requested(open_one_jaw, held, 0.01).tolist() == [True]
    assert donor_opening_requested(paired_open, held, 0.01).tolist() == [True]


def test_donor_release_guard_clamps_unqualified_jaw_targets_at_the_actuator(monkeypatch):
    """Exercise the production action term rather than only its pure predicates."""

    from isaaclab_tasks.manager_based.manipulation.needle_pass.mdp import terminations

    class _Asset:
        def __init__(self):
            self.calls = []

        def set_joint_position_target(self, command, joint_ids):
            self.calls.append((command.clone(), joint_ids))

    hold = torch.tensor(((-0.20, 0.01),), dtype=torch.float32)
    machine = SimpleNamespace(
        phase=torch.tensor((int(HandoffPhase.CO_HOLD),) * 3),
        _receiver_engaged=torch.ones(3, dtype=torch.bool),
        _bilateral_contact=lambda loads, normals, engaged: torch.tensor((False, True, False)),
    )
    env = SimpleNamespace()
    action = object.__new__(DonorReleaseGuardedPairedJawJointPositionAction)
    action._env = env
    action._hold_target = hold.expand(3, -1).clone()
    action._joint_ids = [6, 7]
    action._asset = _Asset()
    action._debug_vis_handle = None
    action.cfg = SimpleNamespace(phase_cfg=object(), release_aperture_threshold_rad=0.01)
    action._processed_actions = torch.tensor(((-0.40, 0.01), (-0.40, 0.30), (-0.40, 0.30)), dtype=torch.float32)

    monkeypatch.setattr(terminations, "get_handoff_phase_machine", lambda *_: machine)
    monkeypatch.setattr(
        terminations,
        "jaw_needle_contact_measurements",
        lambda _: (torch.zeros((3, 4)), torch.zeros((3, 4, 3)), torch.zeros((3, 4))),
    )

    action.apply_actions()

    command, joint_ids = action._asset.calls.pop()
    torch.testing.assert_close(command, torch.tensor(((-0.20, 0.01), (-0.40, 0.30), (-0.20, 0.01))))
    assert joint_ids == [6, 7]


def test_receiver_bounds_use_configured_nonidentity_grasp_transform():
    half_angle = np.pi / 4.0
    cfg = HandoffPhaseCfg(
        receiver_relative_position_target_m=(0.01, -0.02, 0.03),
        receiver_relative_orientation_target_wxyz=(np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)),
        receiver_relative_position_limit_m=1.0e-4,
        receiver_relative_orientation_limit_rad=1.0e-4,
    )
    machine = HandoffPhaseMachine(1, "cpu", 0.01, cfg)
    receiver_pose = torch.tensor(((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),), dtype=torch.float32)
    needle_pose = torch.tensor(
        ((0.01, -0.02, 0.03, np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)),), dtype=torch.float32
    )
    measurements = HandoffMeasurements(
        normal_forces_n=torch.zeros((1, 4)),
        reaction_normals_w=torch.tensor(JAW_BODY_REACTION_NORMALS_LOCAL).unsqueeze(0),
        needle_pose_w=needle_pose,
        needle_velocity_w=torch.zeros((1, 6)),
        receiver_pose_w=receiver_pose,
    )
    assert machine._receiver_bounds(measurements).item() is True
    measurements.needle_pose_w[:, 0] += 0.001
    assert machine._receiver_bounds(measurements).item() is False
    measurements.needle_pose_w[:, 0] -= 0.001
    measurements.receiver_pose_w[:, 0] = torch.nan
    assert machine._receiver_bounds(measurements).item() is False


@pytest.mark.parametrize("pose_name", ("needle_pose_w", "receiver_pose_w"))
def test_receiver_bounds_reject_degenerate_input_quaternion(pose_name):
    """A zero input quaternion cannot satisfy the receiver grasp bounds."""

    machine = HandoffPhaseMachine(1, "cpu", 0.01, HandoffPhaseCfg())
    measurements = _measurements(1, torch.zeros((1, 4)))
    getattr(measurements, pose_name)[:, 3:7] = 0.0
    assert machine._receiver_bounds(measurements).tolist() == [False]


def test_reset_write_order_and_single_needle_state_write():
    calls = []
    left_default = torch.tensor(((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, *DONOR_GRASP_JAW_POS),), dtype=torch.float32)
    right_default = torch.tensor(
        ((10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -np.pi / 6.0, np.pi / 6.0),),
        dtype=torch.float32,
    )

    class MockPSM:
        def __init__(self, name, default_joint_pos):
            self.name = name
            self.expected_joint_pos = default_joint_pos
            self.data = SimpleNamespace(default_joint_pos=default_joint_pos)

        def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
            calls.append((self.name, "state"))
            torch.testing.assert_close(joint_pos, self.expected_joint_pos)
            torch.testing.assert_close(joint_vel, torch.zeros_like(joint_pos))

        def set_joint_position_target(self, joint_pos, env_ids):
            calls.append((self.name, "position_target"))
            torch.testing.assert_close(joint_pos, self.expected_joint_pos)

        def set_joint_velocity_target(self, joint_vel, env_ids):
            calls.append((self.name, "velocity_target"))

    class MockNeedle:
        def __init__(self):
            self.data = SimpleNamespace(
                default_root_state=torch.tensor(((0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0),))
            )

        def write_root_pose_to_sim(self, pose, env_ids):
            calls.append(("needle", "pose"))
            torch.testing.assert_close(pose[:, :3], torch.tensor(((1.1, 2.2, 3.3),)))

        def write_root_velocity_to_sim(self, velocity, env_ids):
            calls.append(("needle", "velocity"))
            torch.testing.assert_close(velocity, torch.zeros_like(velocity))

    class MockScene(dict):
        env_origins = torch.tensor(((1.0, 2.0, 3.0),))

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.01,
        common_step_counter=4,
        scene=MockScene(
            left_psm=MockPSM("left", left_default),
            right_psm=MockPSM("right", right_default),
            needle=MockNeedle(),
        ),
    )
    reset_needle_pass_to_default(env, torch.tensor([0]), HandoffPhaseCfg())
    assert calls == [
        ("left", "state"),
        ("left", "position_target"),
        ("left", "velocity_target"),
        ("right", "state"),
        ("right", "position_target"),
        ("right", "velocity_target"),
        ("needle", "pose"),
        ("needle", "velocity"),
    ]


def test_contact_phase_sequence_counterfactual_and_partial_reset():
    cfg = HandoffPhaseCfg(
        donor_dwell_s=0.02,
        co_hold_dwell_s=0.02,
        receiver_only_dwell_s=0.02,
        retained_lift_dwell_s=0.02,
        required_lift_delta_z_m=0.01,
    )
    machine = HandoffPhaseMachine(2, "cpu", 0.01, cfg)
    machine.reset(torch.tensor((0, 1)), torch.zeros(2), step_token=0)
    donor = torch.tensor(((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
    for token in (1, 2):
        machine.advance(_measurements(2, donor), token)
    assert machine.phase.tolist() == [HandoffPhase.DONOR_HOLD, HandoffPhase.INITIAL]
    for token in (3, 4):
        machine.advance(_measurements(2, torch.tensor(((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)))), token)
    assert machine.phase.tolist() == [HandoffPhase.CO_HOLD, HandoffPhase.INITIAL]
    for token in (5, 6):
        machine.advance(_measurements(2, torch.tensor(((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)))), token)
    assert machine.phase.tolist() == [HandoffPhase.RECEIVER_ONLY_HOLD, HandoffPhase.INITIAL]
    for token in (7, 8):
        machine.advance(
            _measurements(2, torch.tensor(((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0))), needle_z=0.02),
            token,
        )
    assert machine.phase.tolist() == [HandoffPhase.RETAINED_LIFT, HandoffPhase.INITIAL]

    machine.reset(torch.tensor([0]), torch.tensor([0.02]), step_token=8)
    assert machine.phase.tolist() == [HandoffPhase.INITIAL, HandoffPhase.INITIAL]
    machine.advance(_measurements(2, torch.zeros((2, 4))), step_token=8)
    assert machine._donor_counter.tolist() == [0, 0]


def test_receiver_only_ownership_survives_a_transient_bounds_excursion():
    """A live bilateral recipient grasp must not roll back on transient motion."""

    cfg = HandoffPhaseCfg(
        donor_dwell_s=0.01,
        co_hold_dwell_s=0.01,
        receiver_only_dwell_s=0.01,
        retained_lift_dwell_s=0.01,
        required_lift_delta_z_m=0.01,
    )
    machine = HandoffPhaseMachine(1, "cpu", 0.01, cfg)
    machine.reset(torch.tensor([0]), torch.tensor([0.0]), step_token=0)
    donor = torch.tensor(((1.0, 1.0, 0.0, 0.0),))
    cohold = torch.tensor(((1.0, 1.0, 1.0, 1.0),))
    receiver = torch.tensor(((0.0, 0.0, 1.0, 1.0),))
    machine.advance(_measurements(1, donor), step_token=1)
    machine.advance(_measurements(1, cohold), step_token=2)
    machine.advance(_measurements(1, receiver), step_token=3)
    assert machine.phase.item() == HandoffPhase.RECEIVER_ONLY_HOLD

    moving_receiver = _measurements(1, receiver, needle_z=0.02)
    moving_receiver.needle_velocity_w[:, 0] = cfg.maximum_linear_velocity_m_s * 2.0
    assert machine._receiver_bounds(moving_receiver).item() is False
    machine.advance(moving_receiver, step_token=4)
    assert machine.phase.item() == HandoffPhase.RECEIVER_ONLY_HOLD
    assert machine._retained_lift_counter.item() == 0

    machine.advance(_measurements(1, torch.zeros((1, 4))), step_token=5)
    assert machine.phase.item() == HandoffPhase.INITIAL


def test_contact_dwell_is_consecutive_across_rollbacks():
    cfg = HandoffPhaseCfg(
        donor_dwell_s=0.02,
        co_hold_dwell_s=0.02,
        receiver_only_dwell_s=0.02,
        retained_lift_dwell_s=0.02,
    )
    machine = HandoffPhaseMachine(1, "cpu", 0.01, cfg)
    machine.reset(torch.tensor([0]), torch.tensor([0.0]), step_token=0)
    donor = torch.tensor(((1.0, 1.0, 0.0, 0.0),))
    cohold = torch.tensor(((1.0, 1.0, 1.0, 1.0),))
    receiver = torch.tensor(((0.0, 0.0, 1.0, 1.0),))

    machine.advance(_measurements(1, donor), 1)
    machine.advance(_measurements(1, donor), 2)
    machine.advance(_measurements(1, cohold), 3)
    assert machine._co_hold_counter.item() == 1
    machine.advance(_measurements(1, receiver), 4)
    assert machine.phase.item() == HandoffPhase.INITIAL
    assert machine._co_hold_counter.item() == 0

    machine.advance(_measurements(1, cohold), 5)
    assert machine.phase.item() == HandoffPhase.INITIAL
    machine.advance(_measurements(1, cohold), 6)
    assert machine.phase.item() == HandoffPhase.DONOR_HOLD
    machine.advance(_measurements(1, cohold), 7)
    assert machine.phase.item() == HandoffPhase.DONOR_HOLD
    machine.advance(_measurements(1, cohold), 8)
    assert machine.phase.item() == HandoffPhase.CO_HOLD

    machine.advance(_measurements(1, donor), 9)
    assert machine.phase.item() == HandoffPhase.DONOR_HOLD
    assert machine._co_hold_counter.item() == 0
    machine.advance(_measurements(1, cohold), 10)
    assert machine.phase.item() == HandoffPhase.DONOR_HOLD
    machine.advance(_measurements(1, cohold), 11)
    assert machine.phase.item() == HandoffPhase.CO_HOLD


def test_contact_projection_order_sign_and_pose_shape():
    assert JAW_BODY_REACTION_NORMALS_LOCAL == (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    sensors = {}
    expected_loads = torch.tensor((1.0, 2.0, 3.0, 4.0))
    for name, normal, load in zip(
        JAW_CONTACT_SENSOR_NAMES,
        JAW_BODY_REACTION_NORMALS_LOCAL,
        expected_loads,
        strict=True,
    ):
        force = torch.tensor(normal, dtype=torch.float32) * load
        sensors[name] = SimpleNamespace(
            data=SimpleNamespace(
                force_matrix_w=force.reshape(1, 1, 1, 3),
                quat_w=torch.tensor((1.0, 0.0, 0.0, 0.0)).reshape(1, 1, 4),
            )
        )
    env = SimpleNamespace(num_envs=1, scene=SimpleNamespace(sensors=sensors))
    loads, normals, force_vectors = jaw_needle_contact_measurements(env)
    torch.testing.assert_close(loads[0], expected_loads)
    torch.testing.assert_close(normals[0], torch.tensor(JAW_BODY_REACTION_NORMALS_LOCAL))
    torch.testing.assert_close(
        force_vectors[0], torch.tensor(JAW_BODY_REACTION_NORMALS_LOCAL) * expected_loads[:, None]
    )

    for sensor in sensors.values():
        sensor.data.force_matrix_w *= -1.0
    loads, _, _ = jaw_needle_contact_measurements(env)
    torch.testing.assert_close(loads, torch.zeros_like(loads))


def test_configured_device_homes_are_finite_and_distinct():
    assert np.isfinite(LEFT_TOOL_HOME_POS_W).all()
    assert np.isfinite(RIGHT_TOOL_HOME_POS_W).all()
    assert LEFT_TOOL_HOME_POS_W != RIGHT_TOOL_HOME_POS_W
