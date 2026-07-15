# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualization regressions for the Franka Pour task."""

from __future__ import annotations

import os

import pytest

_RUNTIME_AVAILABLE = bool(os.environ.get("EXP_PATH"))
_RUNTIME_UNAVAILABLE_REASON = "Isaac Sim runtime is unavailable because EXP_PATH is not set."
_TEST_DEVICE = os.environ.get("ISAACLAB_TEST_DEVICE", "cuda:0")

if _RUNTIME_AVAILABLE:
    from isaaclab.app import AppLauncher

    # Launch Kit before importing simulation-dependent modules.
    app_launcher = AppLauncher(headless=True, enable_cameras=True, device=_TEST_DEVICE)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import newton
    import numpy as np
    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg

    import usdrt
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourEnvCfg

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.newton_ci]

_TASK_ID = "Isaac-Pour-Franka-v0"
_RESET_DATASET_PLAY_TASK_ID = "Isaac-Pour-Franka-Reset-Dataset-Play-v0"
_SCENE_PARTITION_ENV_VAR = "ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION"


def test_franka_pour_viewer_frames_first_environment():
    """The task viewer should interpret its camera pose relative to environment zero."""
    cfg = FrankaPourEnvCfg()

    assert cfg.viewer.origin_type == "env"
    assert cfg.viewer.env_index == 0


def _make_visualization_cfg():
    cfg = parse_env_cfg(_TASK_ID, device=_TEST_DEVICE, num_envs=2)
    cfg.seed = 37
    cfg.curriculum_start_stage = cfg.curriculum_stage_names.index("full")
    cfg.curriculum_freeze = True
    cfg.scene.env_spacing = 2.5
    cfg.decimation = 1
    cfg.physics_substeps = 2
    cfg.mpm_iterations = 2
    cfg.use_cuda_graph = False
    cfg.sim.render_interval = 1
    # Keep the eager double-buffered manager on an even substep count, matching
    # the production configuration's stable public state bindings.
    cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(headless=True, randomly_sample_visible_envs=False),
        NewtonVisualizerCfg(
            headless=True,
            show_particles=True,
            enable_shadows=False,
            enable_sky=False,
            randomly_sample_visible_envs=False,
        ),
    ]
    return cfg


def _assert_unpartitioned(prim, attribute_name: str) -> None:
    attribute = prim.GetAttribute(attribute_name)
    assert not attribute.IsValid() or not attribute.HasAuthoredValueOpinion(), (
        f"Unexpected authored {attribute_name!r} on {prim.GetPath()}."
    )


def _shape_matches(model, label_fragment: str) -> list[tuple[int, int, bool]]:
    visible_flag = int(newton.ShapeFlags.VISIBLE)
    flags = model.shape_flags.numpy()
    worlds = model.shape_world.numpy()
    matches = [
        (shape_id, int(worlds[shape_id]), bool(int(flags[shape_id]) & visible_flag))
        for shape_id, label in enumerate(model.shape_label)
        if label_fragment in str(label)
    ]
    assert matches, f"No Newton shapes matched {label_fragment!r}."
    return matches


def _assert_shape_distribution(
    model, label_fragment: str, *, visible: bool, expected_per_world: dict[int, int]
) -> None:
    matches = _shape_matches(model, label_fragment)
    actual_per_world = {
        world: sum(match_world == world for _, match_world, _ in matches) for world in expected_per_world
    }
    assert actual_per_world == expected_per_world, (label_fragment, matches)
    assert all(match_visible is visible for _, _, match_visible in matches), (label_fragment, matches)


def _pose_xyzw_to_fabric_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert an XYZ + XYZW pose to Fabric's row-vector matrix convention."""
    position = pose[:3]
    x, y, z, w = pose[3:7] / np.linalg.norm(pose[3:7])
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.T
    matrix[3, :3] = position
    return matrix


def _fabric_world_matrix(path: str) -> np.ndarray:
    """Read a fully materialized Fabric hierarchy matrix without creating a view."""
    stage = NewtonManager._usdrt_stage
    assert stage is not None
    prim = stage.GetPrimAtPath(usdrt.Sdf.Path(path))
    assert prim.IsValid(), path
    xformable = usdrt.Rt.Xformable(prim)
    assert xformable.GetFabricHierarchyLocalMatrixAttr().IsValid(), path
    attribute = xformable.GetFabricHierarchyWorldMatrixAttr()
    assert attribute.IsValid(), path
    return np.asarray(attribute.Get(), dtype=np.float64)


def _assert_descendant_follows_body(
    usd_stage, xform_cache: UsdGeom.XformCache, body_path: str, descendant_path: str, expected_body: np.ndarray
) -> None:
    """Compare a descendant against its authored body-relative transform."""
    body_prim = usd_stage.GetPrimAtPath(body_path)
    descendant_prim = usd_stage.GetPrimAtPath(descendant_path)
    assert body_prim.IsValid(), body_path
    assert descendant_prim.IsValid(), descendant_path
    authored_body = np.asarray(xform_cache.GetLocalToWorldTransform(body_prim), dtype=np.float64)
    authored_descendant = np.asarray(xform_cache.GetLocalToWorldTransform(descendant_prim), dtype=np.float64)
    body_relative = authored_descendant @ np.linalg.inv(authored_body)
    expected_descendant = body_relative @ expected_body
    np.testing.assert_allclose(
        _fabric_world_matrix(descendant_path),
        expected_descendant,
        rtol=0.0,
        atol=1.0e-5,
        err_msg=descendant_path,
    )


def _assert_visual_descendants_follow_fabric_bodies(task) -> None:
    """Check body roots, visual instance roots, and proxy meshes in raw Fabric."""
    usd_stage = sim_utils.get_current_stage()
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    robot_poses = task.scene["robot"].data.body_link_pose_w.torch.detach().cpu().numpy()

    for env_id in range(task.num_envs):
        robot_path = f"/World/envs/env_{env_id}/Robot"
        for body_id, body_name in enumerate(task.scene["robot"].body_names):
            body_path = f"{robot_path}/{body_name}"
            expected_body = _pose_xyzw_to_fabric_matrix(robot_poses[env_id, body_id])
            np.testing.assert_allclose(
                _fabric_world_matrix(body_path),
                expected_body,
                rtol=0.0,
                atol=1.0e-5,
                err_msg=body_path,
            )
            _assert_descendant_follows_body(usd_stage, xform_cache, body_path, f"{body_path}/visuals", expected_body)
            _assert_descendant_follows_body(
                usd_stage, xform_cache, body_path, f"{body_path}/visuals/{body_name}", expected_body
            )

        table_path = f"/World/envs/env_{env_id}/Table"
        table_world = _fabric_world_matrix(table_path)
        _assert_descendant_follows_body(usd_stage, xform_cache, table_path, f"{table_path}/Visuals", table_world)
        _assert_descendant_follows_body(
            usd_stage, xform_cache, table_path, f"{table_path}/Visuals/TableGeom", table_world
        )

    for scene_name, prim_name in (("source_cup", "SourceCup"), ("target_cup", "TargetCup")):
        poses = task.scene[scene_name].data.root_link_pose_w.torch.detach().cpu().numpy()
        for env_id in range(task.num_envs):
            body_path = f"/World/envs/env_{env_id}/{prim_name}"
            expected_body = _pose_xyzw_to_fabric_matrix(poses[env_id])
            np.testing.assert_allclose(
                _fabric_world_matrix(body_path),
                expected_body,
                rtol=0.0,
                atol=1.0e-5,
                err_msg=body_path,
            )
            _assert_descendant_follows_body(usd_stage, xform_cache, body_path, f"{body_path}/geometry", expected_body)
            _assert_descendant_follows_body(
                usd_stage, xform_cache, body_path, f"{body_path}/geometry/mesh", expected_body
            )


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_reset_dataset_play_captures_sparse_outer_graph():
    """Sparse reset-dataset playback should capture and replay the complete coupled solve graph."""
    sim_utils.create_new_stage()
    env = None
    try:
        cfg = parse_env_cfg(_RESET_DATASET_PLAY_TASK_ID, device=_TEST_DEVICE, num_envs=1)
        cfg.seed = 37
        cfg.decimation = 1
        cfg.physics_substeps = 1
        cfg.mpm_iterations = 2
        cfg.sim.render_interval = 1
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, randomly_sample_visible_envs=False)]
        env = gym.make(_RESET_DATASET_PLAY_TASK_ID, cfg=cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        actions = torch.zeros((1, task.action_manager.total_action_dim), device=task.device)
        env.step(actions)
        env.step(actions)
        wp.synchronize_device(task.device)

        mpm_solver = NewtonManager._solver.solver("media")
        assert NewtonManager._graph is not None
        assert mpm_solver._use_cuda_graph is True
        assert bool(torch.all(task.state_finite()))
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_kit_and_newton_visualize_both_worlds(monkeypatch: pytest.MonkeyPatch):
    """Both renderers should consume the two spaced worlds without adding another world offset."""
    monkeypatch.delenv(_SCENE_PARTITION_ENV_VAR, raising=False)
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_visualization_cfg())
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None

        # Validate the natural Newton-to-Fabric setup before a view can initialize
        # or otherwise mask missing descendant hierarchy attributes.
        _assert_visual_descendants_follow_fabric_bodies(task)

        env.reset()

        stage = sim_utils.get_current_stage()
        model = NewtonManager.get_model()
        wp.synchronize_device(model.device)
        assert str(model.device) == _TEST_DEVICE

        _assert_visual_descendants_follow_fabric_bodies(task)

        expected_origins = np.array([[1.25, 0.0, 0.0], [-1.25, 0.0, 0.0]], dtype=np.float32)
        origins = task.scene.env_origins.detach().cpu().numpy()
        np.testing.assert_allclose(origins, expected_origins, rtol=0.0, atol=1.0e-6)

        local_positions = {
            "Robot": tuple(task.cfg.scene.robot.init_state.pos),
            "Table": tuple(task.cfg.scene.table.init_state.pos),
            "SourceCup": tuple(task.cfg.cup_reset_pos),
            "TargetCup": tuple(task.cfg.target_cup_reset_pos),
        }
        for env_id, origin in enumerate(origins):
            env_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
            assert env_root.IsValid()
            assert UsdGeom.Imageable(env_root).ComputeVisibility() == UsdGeom.Tokens.inherited
            _assert_unpartitioned(env_root, "primvars:omni:scenePartition")

            for asset_name, expected_local_position in local_positions.items():
                asset_path = f"/World/envs/env_{env_id}/{asset_name}"
                asset_prim = stage.GetPrimAtPath(asset_path)
                assert asset_prim.IsValid(), asset_path
                local_position, _ = sim_utils.resolve_prim_pose(asset_prim, ref_prim=env_root)
                world_position, _ = sim_utils.resolve_prim_pose(asset_prim)
                np.testing.assert_allclose(local_position, expected_local_position, rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(
                    world_position,
                    origin + np.asarray(expected_local_position),
                    rtol=0.0,
                    atol=1.0e-6,
                )

            for cup_name in ("SourceCup", "TargetCup"):
                mesh_path = f"/World/envs/env_{env_id}/{cup_name}/geometry/mesh"
                mesh = UsdGeom.Mesh.Get(stage, mesh_path)
                assert mesh.GetPrim().IsValid(), mesh_path
                assert UsdGeom.Imageable(mesh).ComputeVisibility() == UsdGeom.Tokens.inherited
            assert not stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Cup").IsValid()
            assert not stage.GetPrimAtPath(f"/World/envs/env_{env_id}/SpillFloor").IsValid()

        source_positions = task.scene["source_cup"].data.root_link_pose_w.torch[:, :3].detach().cpu().numpy()
        target_positions = task.scene["target_cup"].data.root_link_pose_w.torch[:, :3].detach().cpu().numpy()
        np.testing.assert_allclose(
            source_positions,
            origins + np.asarray(task.cfg.cup_reset_pos),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            target_positions,
            origins + np.asarray(task.cfg.target_cup_reset_pos),
            rtol=0.0,
            atol=1.0e-6,
        )
        body_labels = [str(label) for label in model.body_label]
        for env_id in (0, 1):
            for body_name in ("SourceCup", "TargetCup", "SpillFloor"):
                expected_label = f"/World/envs/env_{env_id}/{body_name}"
                assert body_labels.count(expected_label) == 1, expected_label
            assert f"/World/envs/env_{env_id}/TargetCupRigid" not in body_labels
            assert f"/World/envs/env_{env_id}/Cup" not in body_labels

        assert {world for _, world, _ in _shape_matches(model, "/Robot/")} == {0, 1}
        _assert_shape_distribution(model, "/SourceCup/geometry/mesh", visible=True, expected_per_world={0: 1, 1: 1})
        _assert_shape_distribution(model, "/TargetCup/geometry/mesh", visible=True, expected_per_world={0: 1, 1: 1})
        _assert_shape_distribution(
            model, "/SourceCup/geometry/grasp_proxy", visible=False, expected_per_world={0: 1, 1: 1}
        )
        table_shapes = _shape_matches(model, "/Table/Collisions/Cube")
        table_shape_distribution = {
            (world, visible): sum(
                match_world == world and match_visible is visible for _, match_world, match_visible in table_shapes
            )
            for world in (0, 1)
            for visible in (False, True)
        }
        assert table_shape_distribution == {
            (0, False): 1,
            (0, True): 1,
            (1, False): 1,
            (1, True): 1,
        }, table_shapes
        grasp_proxy_shape_ids = [
            shape_id for shape_id, _, _ in _shape_matches(model, "/SourceCup/geometry/grasp_proxy")
        ]
        for shape_id in grasp_proxy_shape_ids:
            np.testing.assert_allclose(
                model.shape_scale.numpy()[shape_id],
                task.cfg.cup_grasp_box_half,
                rtol=0.0,
                atol=1.0e-6,
            )
        _assert_shape_distribution(model, "/ParticleCollider", visible=False, expected_per_world={0: 2, 1: 2})
        _assert_shape_distribution(model, "/TargetCup/Collision", visible=False, expected_per_world={0: 1, 1: 1})
        _assert_shape_distribution(model, "/SpillFloor/Collision", visible=False, expected_per_world={0: 1, 1: 1})
        assert set(model.particle_world.numpy().tolist()) == {0, 1}

        point_paths = {
            prim.GetPath().pathString
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Points) and prim.GetPath().pathString.startswith("/World/Visuals/MPMParticles/")
        }
        assert len(point_paths) == 2
        assert {path.rsplit("/", 1)[-1] for path in point_paths} == {"env_0", "env_1"}
        point_path_by_env = {path.rsplit("/", 1)[-1]: path for path in point_paths}

        cameras = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
        assert cameras
        for camera in cameras:
            _assert_unpartitioned(camera, "omni:scenePartition")

        kit_visualizers = [visualizer for visualizer in task.sim.visualizers if isinstance(visualizer, KitVisualizer)]
        newton_visualizers = [
            visualizer for visualizer in task.sim.visualizers if isinstance(visualizer, NewtonVisualizer)
        ]
        assert len(kit_visualizers) == 1
        assert len(newton_visualizers) == 1
        assert kit_visualizers[0].cfg.max_visible_envs is None
        assert kit_visualizers[0].get_visualized_env_ids() is None

        newton_visualizer = newton_visualizers[0]
        assert newton_visualizer.cfg.max_visible_envs is None
        assert newton_visualizer.get_visualized_env_ids() is None
        # NewtonVisualizer has no public accessor for the native viewer.
        viewer = newton_visualizer._viewer
        assert viewer is not None
        assert model.world_count == 2
        assert viewer._visible_worlds is None
        assert viewer._visible_worlds_mask is None
        np.testing.assert_array_equal(viewer.world_offsets.numpy(), np.zeros((2, 3), dtype=np.float32))
        assert viewer.show_particles is True

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        env.step(actions)
        wp.synchronize_device(model.device)

        _assert_visual_descendants_follow_fabric_bodies(task)

        particle_world = model.particle_world.numpy()
        particle_q = NewtonManager.get_state_0().particle_q.numpy()
        for env_id in range(task.num_envs):
            points = UsdGeom.Points.Get(stage, point_path_by_env[f"env_{env_id}"]).GetPointsAttr().Get()
            actual = np.asarray(points, dtype=np.float32)
            expected = particle_q[particle_world == env_id]
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)
    finally:
        if env is not None:
            env.close()
