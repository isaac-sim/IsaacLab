# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless runtime integration for the reset-dataset Franka Pour task."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_RUNTIME_UNAVAILABLE_REASON = "Isaac Sim runtime is unavailable because EXP_PATH is not set."
_RUNTIME_AVAILABLE = bool(os.environ.get("EXP_PATH"))

if _RUNTIME_AVAILABLE:
    from isaaclab.app import AppLauncher

    # Launch Kit before importing simulation-dependent modules.
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import newton
    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverMuJoCo

    import isaaclab.sim as sim_utils

    from isaaclab_tasks.contrib.franka_pour.media import build_media_object_cfg
    from isaaclab_tasks.contrib.franka_pour.pour_env import (
        _MJWARP_SOLREF_MODE_FORCE_SPACE,
        FrankaPourEnv,
        _set_mjwarp_force_space_solref_mode,
    )
    from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import (
        MPM_ENTRY,
        FrankaPourResetDatasetEnvCfg,
        _reset_dataset_task_contract,
    )
    from isaaclab_tasks.contrib.franka_pour.reset_dataset_io import (
        FRANKA_POUR_RESET_DATASET_FORMAT,
        FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
        reset_dataset_content_digest,
        reset_dataset_digest,
    )

pytestmark = pytest.mark.isaacsim_ci

SOLREF_MODE_FORCE_SPACE = _MJWARP_SOLREF_MODE_FORCE_SPACE if _RUNTIME_AVAILABLE else 0
_PHYSICS_SHA256 = "1" * 64
_SOURCE_SHA256 = "2" * 64


def _require_cuda() -> None:
    """Require a CUDA device for the Franka Pour runtime integration."""
    if not wp.is_cuda_available():
        pytest.skip("Franka Pour runtime integration requires a CUDA device.")


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_task_force_space_mode_matches_newton_main_contract() -> None:
    """The task-local value must still select Newton's documented FORCE_SPACE behavior."""
    _require_cuda()
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    body_id = builder.add_body(mass=1.0)
    shape_id = builder.add_shape_sphere(body_id, radius=0.1)
    _set_mjwarp_force_space_solref_mode(builder, shape_id)
    model = builder.finalize(device="cuda:0")

    with pytest.warns(UserWarning, match="SOLREF_MODE_FORCE_SPACE"):
        SolverMuJoCo(model, use_mujoco_contacts=True, njmax=20, iterations=1)


def _build_reset_dataset(cfg: FrankaPourResetDatasetEnvCfg) -> dict:
    """Build a minimal production-shaped artifact for runtime integration tests."""
    state_count = 2
    task_contract = _reset_dataset_task_contract(cfg)
    sampler_cfg = {
        "sampling_profile": cfg.reset_dataset_expected_sampling_profile,
        "grasp_side_ids": cfg.reset_dataset_expected_grasp_side_ids,
    }
    measured_result = {
        "passed": True,
        "source_content_sha256": _SOURCE_SHA256,
        "physics_contract_sha256": _PHYSICS_SHA256,
    }
    metadata = {
        "state_count": state_count,
        "joint_names": tuple(f"panda_joint{index}" for index in range(1, 8))
        + ("panda_finger_joint1", "panda_finger_joint2"),
        "frame": "environment",
        "quaternion_order": "xyzw",
        "particle_solver_state": "fresh_zero",
        "sampler_cfg": sampler_cfg,
        "task_contract": task_contract,
        "static_validation": {
            "policy": "analytic_static_v1",
            "all_rows_statically_validated": True,
            "per_row_mpm_rollout": False,
            "terminal_pour_manifold": {
                "policy": "relative_source_receiver_v1",
                "physics_contract_sha256": _PHYSICS_SHA256,
                "calibration": {
                    "policy": "bounded_terminal_gpu_sweep_v2",
                    "status": "passed",
                    "source_content_sha256": _SOURCE_SHA256,
                    "physics_contract_sha256": _PHYSICS_SHA256,
                    "result_sha256": reset_dataset_digest(measured_result),
                    "measured_result": measured_result,
                },
            },
        },
    }

    arm_q = torch.tensor(cfg.arm_home, dtype=torch.float32).repeat(state_count, 1)
    finger_q = torch.full((state_count, 2), float(cfg.gripper_open_pos), dtype=torch.float32)
    identity_quat = (0.0, 0.0, 0.0, 1.0)
    source_pose = torch.tensor((*cfg.cup_reset_pos, *identity_quat), dtype=torch.float32).repeat(state_count, 1)
    target_pose = torch.tensor((*cfg.target_cup_reset_pos, *identity_quat), dtype=torch.float32).repeat(state_count, 1)
    media_cfg = build_media_object_cfg(cfg, cfg.cup_reset_pos, identity_quat)
    grid = media_cfg.spawn
    lower = torch.tensor(grid.lower, dtype=torch.float32)
    upper = torch.tensor(grid.upper, dtype=torch.float32)
    resolution = torch.ceil(grid.particles_per_cell * (upper - lower) / grid.voxel_size).to(torch.int64)
    cell = (upper - lower) / resolution
    axes = [
        lower[axis] + (torch.arange(int(resolution[axis]), dtype=torch.float32) + 0.5) * cell[axis] for axis in range(3)
    ]
    local_position = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)
    payload = {
        "format": FRANKA_POUR_RESET_DATASET_FORMAT,
        "schema_version": FRANKA_POUR_RESET_DATASET_SCHEMA_VERSION,
        "metadata": metadata,
        "states": {
            "arm_joint_position": arm_q,
            "arm_joint_velocity": torch.zeros((state_count, 7), dtype=torch.float32),
            "finger_joint_position": finger_q,
            "finger_joint_velocity": torch.zeros((state_count, 2), dtype=torch.float32),
            "finger_joint_target": finger_q.clone(),
            "source_root_pose": source_pose,
            "source_root_velocity": torch.zeros((state_count, 6), dtype=torch.float32),
            "target_root_pose": target_pose,
            "target_root_velocity": torch.zeros((state_count, 6), dtype=torch.float32),
            "category": torch.zeros(state_count, dtype=torch.int8),
            "objective": torch.zeros(state_count, dtype=torch.float32),
            "difficulty": torch.zeros(state_count, dtype=torch.float32),
            "particle_layout_id": torch.zeros(state_count, dtype=torch.int32),
        },
        "particle_layouts": {
            "local_position": local_position,
            "local_velocity": torch.zeros_like(local_position),
        },
        "contract_sha256": reset_dataset_digest({"sampler_cfg": sampler_cfg, "task_contract": task_contract}),
    }
    payload["content_sha256"] = reset_dataset_content_digest(payload)
    return payload


def _make_runtime_cfg(
    dataset_path: Path,
    *,
    use_cuda_graph: bool,
) -> FrankaPourResetDatasetEnvCfg:
    """Configure two deterministic worlds backed by a temporary reset artifact."""
    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = 2
    cfg.scene.env_spacing = 2.5
    cfg.seed = 37
    cfg.curriculum_freeze = True
    cfg.use_cuda_graph = use_cuda_graph
    cfg.sim.render_interval = 1
    payload = _build_reset_dataset(cfg)
    torch.save(payload, dataset_path)
    cfg.reset_dataset_path = str(dataset_path)
    cfg.reset_dataset_content_sha256 = payload["content_sha256"]

    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    assert entries[MPM_ENTRY].in_place
    return cfg


def _assert_scene_solver_roles(
    model,
    *,
    rigid_contact_margin: float,
    table_contact_margin: float,
    mpm_collider_margin: float,
    rigid_contact_ke: float,
    rigid_contact_kd: float,
) -> None:
    """Check exact per-world task bodies and solver-only collision roles."""
    body_world = model.body_world.numpy()
    body_mass = model.body_mass.numpy()
    body_inv_mass = model.body_inv_mass.numpy()
    body_flags = model.body_flags.numpy()
    shape_body = model.shape_body.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_margin = model.shape_margin.numpy()
    shape_ke = model.shape_material_ke.numpy()
    shape_kd = model.shape_material_kd.numpy()
    solref_mode = model.mujoco.solref_mode.numpy()
    collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
    collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    visible = int(newton.ShapeFlags.VISIBLE)

    for world in range(2):
        bodies_by_name: dict[str, list[int]] = {}
        for body_id, label in enumerate(model.body_label):
            if int(body_world[body_id]) == world:
                bodies_by_name.setdefault(str(label).rsplit("/", 1)[-1], []).append(body_id)
        for name in ("SourceCup", "TargetCup", "SpillFloor"):
            assert len(bodies_by_name.get(name, [])) == 1, (world, name, bodies_by_name.get(name))
        assert "TargetCupRigid" not in bodies_by_name

        target_body = bodies_by_name["TargetCup"][0]
        assert int(body_flags[target_body]) & int(newton.BodyFlags.KINEMATIC)
        assert float(body_mass[target_body]) == 0.0
        assert float(body_inv_mass[target_body]) == 0.0

        expected_shapes = (
            ("SourceCup", "/SourceCup/geometry/grasp_proxy", True, False, False, rigid_contact_margin),
            ("SourceCup", "/SourceCup/geometry/bowl", False, True, True, mpm_collider_margin),
            ("TargetCup", "/TargetCup/geometry/bowl", False, True, True, mpm_collider_margin),
            ("TargetCup", "/TargetCup/geometry/rigid_collider", True, False, False, rigid_contact_margin),
            ("SpillFloor", "/SpillFloor/Collision", False, True, False, mpm_collider_margin),
        )
        for body_name, suffix, rigid, particles, is_visible, expected_margin in expected_shapes:
            body_id = bodies_by_name[body_name][0]
            matches = [
                shape_id
                for shape_id, label in enumerate(model.shape_label)
                if int(shape_body[shape_id]) == body_id and str(label).endswith(suffix)
            ]
            assert len(matches) == 1, (world, body_name, matches)
            shape_id = matches[0]
            flags = int(shape_flags[shape_id])
            assert bool(flags & collide_shapes) is rigid
            assert bool(flags & collide_particles) is particles
            assert bool(flags & visible) is is_visible
            assert float(shape_margin[shape_id]) == pytest.approx(expected_margin)
            if suffix == "/SourceCup/geometry/grasp_proxy":
                assert float(shape_ke[shape_id]) == pytest.approx(rigid_contact_ke)
                assert float(shape_kd[shape_id]) == pytest.approx(rigid_contact_kd)
                assert int(solref_mode[shape_id]) == SOLREF_MODE_FORCE_SPACE

        table_shapes = [
            shape_id
            for shape_id, label in enumerate(model.shape_label)
            if int(shape_body[shape_id]) >= 0
            and int(body_world[int(shape_body[shape_id])]) == world
            and str(label).endswith("/Table/Collisions/Cube")
            and int(shape_flags[shape_id]) & collide_shapes
        ]
        assert len(table_shapes) == 1, (world, table_shapes)
        table_shape = table_shapes[0]
        assert float(shape_margin[table_shape]) == pytest.approx(table_contact_margin)
        assert float(shape_ke[table_shape]) == pytest.approx(rigid_contact_ke)
        assert float(shape_kd[table_shape]) == pytest.approx(rigid_contact_kd)
        assert int(solref_mode[table_shape]) == SOLREF_MODE_FORCE_SPACE


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_reset_dataset_cuda_graph_replays_after_selective_reset(tmp_path: Path) -> None:
    """A masked dataset reset preserves the other coupled world and captured graph."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    try:
        env = FrankaPourEnv(_make_runtime_cfg(tmp_path / "reset_dataset.pt", use_cuda_graph=True))
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0
        for _ in range(4):
            env.step(actions)

        graph = NewtonManager._graph
        assert task.cfg.sim.physics.use_cuda_graph
        assert graph is not None
        assert task.sim.physics_manager.handles_decimation()
        mpm_solver = NewtonManager._solver.solver(MPM_ENTRY)
        mpm_solver.check_status()

        world_1_q = task._robot.data.joint_pos.torch[1].clone()
        world_1_cup = task._source_cup.data.root_link_pose_w.torch[1].clone()
        world_1_media = task._media.data.particle_pos_w.torch[1].clone()
        task.reset_pour_scene(torch.tensor([0], device=task.device, dtype=torch.long))
        wp.synchronize_device(NewtonManager.get_model().device)

        torch.testing.assert_close(task._robot.data.joint_pos.torch[1], world_1_q, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._source_cup.data.root_link_pose_w.torch[1], world_1_cup, rtol=0.0, atol=0.0)
        torch.testing.assert_close(task._media.data.particle_pos_w.torch[1], world_1_media, rtol=0.0, atol=0.0)

        observations, _, _, _, _ = env.step(actions)
        mpm_solver.check_status()
        assert NewtonManager._graph is graph
        assert bool(torch.all(task.state_finite()))
        assert bool(torch.all(task.particles_in_workspace()))
        for group in ("policy", "media", "privileged"):
            assert bool(torch.isfinite(observations[group]).all())
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_scene_uses_public_assets_and_expected_solver_roles(tmp_path: Path) -> None:
    """Resolved public cup assets keep their intended rigid and MPM collision roles."""
    _require_cuda()
    sim_utils.create_new_stage()
    env = None
    caller_cfg = _make_runtime_cfg(tmp_path / "reset_dataset.pt", use_cuda_graph=False)
    assert caller_cfg.scene.source_cup is None
    assert caller_cfg.scene.target_cup is None
    assert caller_cfg.scene.media is None
    try:
        env = FrankaPourEnv(caller_cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        assert caller_cfg.scene.source_cup is None
        assert caller_cfg.scene.target_cup is None
        assert caller_cfg.scene.media is None
        assert task.cfg is not caller_cfg

        source_cup = task.scene["source_cup"]
        target_cup = task.scene["target_cup"]
        assert source_cup.num_instances == 2
        assert target_cup.num_instances == 2
        source_pose_e = source_cup.data.root_link_pose_w.torch.clone()
        source_pose_e[:, :3] -= task.scene.env_origins
        target_pose_e = target_cup.data.root_link_pose_w.torch.clone()
        target_pose_e[:, :3] -= task.scene.env_origins
        torch.testing.assert_close(task.cup_pose_e(), source_pose_e)
        torch.testing.assert_close(task.target_pose_e(), target_pose_e)

        model = NewtonManager.get_model()
        _assert_scene_solver_roles(
            model,
            rigid_contact_margin=task.cfg.collider_margin,
            table_contact_margin=0.003 - task.cfg.collider_margin,
            mpm_collider_margin=task.cfg.mpm_collider_margin,
            rigid_contact_ke=task.cfg.grasp_contact_ke,
            rigid_contact_kd=task.cfg.grasp_contact_kd,
        )
        assert model.particle_max_velocity == pytest.approx(task.cfg.particle_max_velocity)

        target_pose_before = target_cup.data.root_link_pose_w.torch.clone()
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0
        env.step(actions)
        wp.synchronize_device(model.device)
        torch.testing.assert_close(target_cup.data.root_link_pose_w.torch, target_pose_before, rtol=0.0, atol=0.0)
    finally:
        if env is not None:
            env.close()
