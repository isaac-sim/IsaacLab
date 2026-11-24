# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical reset and task availability for bounded tabletop worlds, without downloaded assets."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import shapely
import torch
import trimesh
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from newton.usd import get_mesh

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import InclusionSet
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tasks.contrib.heterogeneous_tasks import mdp
from isaaclab_tasks.contrib.heterogeneous_tasks.heterogeneous_tasks_env_cfg import TableCfg

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _container(path: Path):
    """An actual open box: floor and four convex collision meshes, not a solid bounding box."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/box").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.2)
    for i, (half, center) in enumerate(
        (
            ((0.1, 0.1, 0.005), (0, 0, 0.005)),
            ((0.005, 0.1, 0.04), (-0.095, 0, 0.04)),
            ((0.005, 0.1, 0.04), (0.095, 0, 0.04)),
            ((0.09, 0.005, 0.04), (0, -0.095, 0.04)),
            ((0.09, 0.005, 0.04), (0, 0.095, 0.04)),
        )
    ):
        points = [
            tuple(center[j] + sign[j] * half[j] for j in range(3)) for sign in itertools.product((-1, 1), repeat=3)
        ]
        mesh = UsdGeom.Mesh.Define(stage, f"/box/hull_{i}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().Save()


def _table(path: Path):
    """A real mesh table with a central opening that a bounding rectangle would incorrectly fill."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/table").GetPrim()
    stage.SetDefaultPrim(root)
    vertices, indices = [], []
    faces = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    for half, center in (
        ((0.18, 0.5, 0.025), (-0.27, 0, 0)),
        ((0.18, 0.5, 0.025), (0.27, 0, 0)),
        ((0.09, 0.2, 0.025), (0, -0.3, 0)),
        ((0.09, 0.2, 0.025), (0, 0.3, 0)),
    ):
        offset = len(vertices)
        vertices.extend(
            tuple(center[j] + sign[j] * half[j] for j in range(3)) for sign in itertools.product((-1, 1), repeat=3)
        )
        indices.extend(index + offset for index in faces)
    mesh = UsdGeom.Mesh.Define(stage, "/table/surface")
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([4] * 24)
    mesh.CreateFaceVertexIndicesAttr(indices)
    stage.GetRootLayer().Save()


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    gripper = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/gripper",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.24)),
    )
    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/[^/]+", filter_prim_paths_expr=["{ENV_REGEX_NS}/[^/]+"], update_period=0.0
    )


@configclass
class _ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        gripper = ObservationTermCfg(func=isaaclab_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("gripper")})

    policy = PolicyCfg()


@configclass
class _ActionsCfg:
    pass


@configclass
class _EnvCfg(ManagerBasedRLEnvCfg):
    scene = _SceneCfg(num_envs=4, env_spacing=3.0)
    sim = sim_utils.SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
        ),
    )
    observations = _ObservationsCfg()
    actions = _ActionsCfg()
    rewards = {}
    events = {}
    terminations = {"success": TerminationTermCfg(func=mdp.task_success)}
    tasks = {
        "on_plate": {"target": "plate", "relation": "on", "instruction": "Place {object} on {target}."},
        "in_box": {"target": "container", "relation": "in", "instruction": "Put {object} in {target}."},
    }
    decimation = 1
    episode_length_s = 60.0


@pytest.fixture(params=test_devices(DeviceScope.CUDA))
def env(request, tmp_path):
    path = tmp_path / "box.usda"
    _container(path)
    plate_path = tmp_path / "plate.usda"
    stage = Usd.Stage.CreateNew(str(plate_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/plate").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateCenterOfMassAttr((0.065, -0.035, 0.046))
    cube = UsdGeom.Cube.Define(stage, "/plate/collider")
    UsdGeom.Xformable(cube).AddTranslateOp().Set((0.04, -0.02, 0.04))
    UsdGeom.Xformable(cube).AddScaleOp().Set((0.11, 0.1, 0.01))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    # A ledge along the surface that carries objects, 3.9 cm above the table: the top of the plate's bounds is
    # above that surface, and an object on the table fits under the ledge, inside the plate's bounds.
    ledge = UsdGeom.Cube.Define(stage, "/plate/ledge")
    UsdGeom.Xformable(ledge).AddTranslateOp().Set((0.04, 0.11, 0.074))
    UsdGeom.Xformable(ledge).AddScaleOp().Set((0.11, 0.03, 0.005))
    UsdPhysics.CollisionAPI.Apply(ledge.GetPrim())
    stage.GetRootLayer().Save()
    bodies = {
        "small": sim_utils.CuboidCfg(size=(0.04, 0.04, 0.04)),
        "thin": sim_utils.CuboidCfg(size=(0.04, 0.04, 0.01)),
        "large": sim_utils.CuboidCfg(size=(0.3, 0.25, 0.06)),
        "plate": sim_utils.UsdFileCfg(usd_path=str(plate_path)),
        "Box_A01": sim_utils.UsdFileCfg(usd_path=str(path)),
        "__empty__": sim_utils.SphereCfg(radius=0.001, visible=False),
    }
    worlds = (
        ("small", "plate", "thin"),
        ("small", "small", "plate"),
        ("large", "plate", "__empty__"),
        ("small", "Box_A01", "plate"),
    )
    table_path = tmp_path / "table.usda"
    _table(table_path)
    cfg = _EnvCfg(seed=42)
    table = TableCfg(
        prim_path="{ENV_REGEX_NS}/work_table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(table_path),
            func=mdp.spawn_table,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.175), rot=(0.0, 0.0, 0.14943813, 0.98877108)),
        surface_outline=((-0.45, -0.5), (0.45, -0.5), (0.45, 0.5), (-0.45, 0.5)),
        surface_height=0.025,
        surface_holes=(((-0.09, -0.1), (0.09, -0.1), (0.09, 0.1), (-0.09, 0.1)),),
    )
    tables = (
        table,
        table.replace(
            spawn=table.spawn.replace(scale=(1.0, 1.0, 1.4)),
            surface_height=0.035,
            init_state=table.init_state.replace(pos=(0.6, 0.0, 0.165)),
        ),
        table.replace(
            spawn=table.spawn.replace(scale=(0.2, 0.2, 1.0)),
            surface_outline=tuple((0.2 * x, 0.2 * y) for x, y in table.surface_outline),
            surface_holes=tuple(tuple((0.2 * x, 0.2 * y) for x, y in hole) for hole in table.surface_holes),
        ),
    )
    cfg.scene.work_table = table.replace(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[candidate.spawn for candidate in tables],
            static=False,
            random_choice=False,
        )
    )
    cfg.sim.device = request.param
    names = [f"object_{slot}" for slot in range(3)]
    collection = {}
    for slot, name in enumerate(names):
        candidates = []
        for label in dict.fromkeys(row[slot] for row in worlds):
            spawn = bodies[label].copy()
            spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg()
            spawn.mass_props = sim_utils.MassCfg(mass=0.1)
            spawn.collision_props = sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)
            spawn.semantic_tags = [
                ("_catalog_key", label),
                *[("case", str(i)) for i, row in enumerate(worlds) if row[slot] == label],
            ]
            if label != "__empty__":
                spawn.semantic_tags.append(("category", "object"))
            if label in ("plate", "Box_A01"):
                spawn.semantic_tags.append(("category", "plate" if label == "plate" else "container"))
            if label == "plate":
                spawn.semantic_tags.append(("name", "white plate"))
            candidates.append(spawn)
        collection[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=candidates, static=False, random_choice=False),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, -10, 0.5)),
        )
    cfg.scene.objects = RigidObjectCollectionCfg(rigid_objects=collection)
    cfg.scene.clone_cfg.clone_combinations = [InclusionSet(assets=["objects"], match_tags=["case"])]
    cfg.events = {
        "reset": EventTermCfg(func=isaaclab_mdp.reset_scene_to_default, mode="reset"),
        "place_objects": EventTermCfg(
            func=mdp.reset_tabletop,
            mode="reset",
            params={
                "objects": SceneEntityCfg("objects"),
                "table": SceneEntityCfg("work_table"),
                "table_variants": tables,
                "workspace": ((0.25, -0.35, 0.1), (0.9, 0.35, 0.6)),
                "gripper_body": "gripper",
                "clearance": 0.005,
                "max_attempts": 256,
                "graspable_size": (0.02, 0.085),
            },
        ),
    }
    env = ManagerBasedRLEnv(cfg)
    try:
        yield env
    finally:
        env.close()


def test_reset_intersects_workspace_and_table_and_binds_only_feasible_tasks(env):
    env.reset()
    state = env.tabletop
    container_world = next(i for i, names in enumerate(state.object_names) if "Box_A01" in names)
    large_world = next(i for i, names in enumerate(state.object_names) if "large" in names)
    # World 0 also holds an object that fits the plate but is too thin for the gripper. The two objects of
    # world 1 share a name, so one instruction stands for both.
    assert [len(state.available_tasks[i]) for i in (0, 1, large_world, container_world)] == [1, 1, 0, 2]
    assert sorted(state.active.sum(1).tolist()) == [2, 3, 3, 3]
    assert not state.penetrating_worlds().any()
    assert all(surface.area == pytest.approx(0.864, abs=1e-6) for surface in state.surfaces[:2])
    assert state.surfaces[2].area < 0.05
    assert all(len(surface.interiors) == 1 for surface in state.surfaces)
    assert state.instructions[0] and state.instructions[1] and state.instructions[container_world]
    assert state.instructions[large_world] == ""
    with pytest.raises(ValueError, match="Unavailable"):
        state.set_task([large_world], [0])
    poses = state.objects.data.body_link_pose_w.torch
    before = poses.clone()
    for _ in range(3):
        table_before = state.table.mesh_variant_ids.torch.clone()
        env.reset(env_ids=torch.tensor([0, container_world], device=env.device))
        poses = state.objects.data.body_link_pose_w.torch
        torch.testing.assert_close(poses[[1, large_world]], before[[1, large_world]])
        assert not torch.equal(poses[0], before[0])
        table_after = state.table.mesh_variant_ids.torch
        assert (table_after[[0, large_world, container_world]] != 2).all(), "Small tables cannot hold these worlds"
        assert (table_after[[0, container_world]] != table_before[[0, container_world]]).all()
        torch.testing.assert_close(table_after[[1, large_world]], table_before[[1, large_world]])
        torch.testing.assert_close(state.tabletop_heights, torch.full_like(state.tabletop_heights, 0.2))
        corners = quat_apply(poses[..., None, 3:].expand(-1, -1, 8, 4), state.local_corners) + poses[..., None, :3]
        # Contact bounds must land at table + clearance even with mesh and COM offsets from the origin.
        torch.testing.assert_close(
            corners[..., 2].amin(-1)[state.active],
            (state.tabletop_heights[:, None] + 0.005).expand_as(state.active)[state.active],
            atol=1e-6,
            rtol=0,
        )
        local = corners - env.scene.env_origins[:, None, None]
        assert (local[state.active] >= local.new_tensor([0.25, -0.35, 0.1])).all()
        assert (local[state.active] <= local.new_tensor([0.9, 0.35, 0.6])).all()
        table = state.table.data.root_link_pose_w.torch
        table_local = quat_apply_inverse(
            table[:, None, None, 3:].expand(-1, 3, 8, 4), corners - table[:, None, None, :3]
        )
        # Compare the small table in the original mesh coordinates, including its scaled opening.
        scale = torch.where(table_after == 2, 0.2, 1.0)
        table_local[..., :2] /= scale[:, None, None, None]
        assert (table_local[..., :2][state.active] >= table_local.new_tensor([-0.45, -0.5]) + 0.005 - 1e-6).all()
        assert (table_local[..., :2][state.active] <= table_local.new_tensor([0.45, 0.5]) - 0.005 + 1e-6).all()
        hole = shapely.box(-0.09, -0.1, 0.09, 0.1)
        for points in table_local[..., :2][state.active].cpu().numpy():
            assert shapely.MultiPoint(points).convex_hull.disjoint(hole)
        assert not state.penetrating_worlds().any()
    assert not env.termination_manager.compute().any()
    inside = next(i for i, task in enumerate(state.available_tasks[container_world]) if task.family == "in_box")
    state.set_task([0, 1, container_world], [0, 0, inside])
    env.reset()
    # Instructions name objects in plain words: by the asset's name tag, or else by its catalog label without the
    # asset code.
    assert state.instructions[0] == "Place small on white plate."
    assert state.instructions[container_world] == "Put small in box."
    target = state.objects.data.body_link_pose_w.torch

    def place(world: int, offset: tuple[float, float, float], body: int = 0, reference: int | None = None):
        """Move an object of the world to an offset in the frame of the world's target, or of another object."""
        pose = target[world, state.target_ids[world] if reference is None else reference].clone()
        pose[:3] += quat_apply(pose[3:][None], pose.new_tensor([offset]))[0]
        ids = torch.tensor([world], device=env.device)
        state.objects.write_body_pose_to_sim_index(body_poses=pose[None, None], body_ids=[body], env_ids=ids)
        state.objects.write_body_velocity_to_sim_index(
            body_velocities=torch.zeros(1, 1, 6, device=env.device), body_ids=[body], env_ids=ids
        )

    def settle(steps: int) -> list[list[bool]]:
        """Step the simulation and return the success of every world at every step."""
        done = []
        for _ in range(steps):
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            done.append(env.termination_manager.compute().tolist())
        return done

    # The object rests on the plate, below the top of its bounds, and in the box.
    place(0, (0, 0, 0.072))
    place(container_world, (0, 0, 0.031))
    expected = [False] * env.num_envs
    expected[0] = expected[container_world] = True
    assert settle(60)[-1] == expected
    # An object that slides across the plate is not placed until it has come to rest.
    state.objects.write_body_velocity_to_sim_index(
        body_velocities=target.new_tensor([[[0.5, 0, 0, 0, 0, 0]]]),
        body_ids=[0],
        env_ids=torch.tensor([0], device=env.device),
    )
    assert not settle(1)[-1][0]
    assert settle(60)[-1] == expected
    # The plate also carries an object whose corners overhang its edge by 5 mm.
    place(0, (-0.055, 0, 0.072))
    assert settle(100)[-1] == expected
    # An object that stands on the table under the ledge touches the plate inside its bounds, but the table
    # carries it, not the plate.
    place(0, (0.04, 0.11, 0.05))
    assert not any(done[0] for done in settle(60))
    sensor = env.scene["contact"]
    force = sensor.data.normal_force_matrix_w.torch[
        0, sensor.sensor_names.index("object_0"), sensor.filter_object_names.index(state.names[state.target_ids[0]])
    ]
    assert force.norm() >= 0.1
    # The instruction of world 1 names two objects alike: whichever of them rests on the plate counts.
    place(1, (0, 0, 0.072), body=1)
    assert settle(60)[-1][1]
    # An object that rests on another receiver than the one the instruction names does not count.
    place(container_world, (0, 0, 0.072), reference=state.object_names[container_world].index("plate"))
    assert not any(done[container_world] for done in settle(60))
    params = dict(env.cfg.events["place_objects"].params, workspace=((2, 2, 0), (3, 3, 1)))
    with pytest.raises(RuntimeError, match="intersection"):
        state(env, torch.tensor([0], device=env.device), **params)
    # The rectangle lies wholly inside the table's opening, while still inside its outer bounds.
    params = dict(env.cfg.events["place_objects"].params, workspace=((0.58, -0.02, 0.1), (0.62, 0.02, 0.6)))
    with pytest.raises(RuntimeError, match="intersection"):
        state(env, torch.tensor([0], device=env.device), **params)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_robotiq_reopens_without_internal_collision_deformation(device):
    """The actual gripper reopens after closing, while its base still collides with external objects."""
    from isaaclab_tasks.contrib.heterogeneous_tasks.heterogeneous_tasks_env_cfg import (
        CONTACT_CFG,
        VLAHeterogeneousEnvCfg,
    )

    cfg = VLAHeterogeneousEnvCfg(seed=42)
    cfg.sim.device = device
    cfg.scene.num_envs = 1
    for name in ("objects", "work_table", "table_fixture", "contact", "over_shoulder_left_camera", "wrist_cam"):
        setattr(cfg.scene, name, None)
    cfg.scene.clone_cfg.clone_combinations = []
    cfg.scene.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/probe",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=CONTACT_CFG,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, -2, 0.5)),
    )
    cfg.observations.policy.over_shoulder_left_camera = None
    cfg.observations.policy.wrist_cam = None
    cfg.events.place_objects = None
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset()
        robot = env.scene["robot"]
        arm, _ = robot.find_joints(cfg.actions.arm.joint_names, preserve_order=True)
        fingers, _ = robot.find_joints(["finger_joint", ".*inner_finger.*", "right_outer_knuckle_joint"])
        action = torch.ones(1, 8, device=env.device)
        action[:, :7] = robot.data.default_joint_pos.torch[:, arm]
        for command in (1, -1, 1):
            action[:, -1] = command
            for _ in range(30):
                env.step(action)
            assert robot.data.joint_pos.torch.isfinite().all()
            if command > 0:
                assert robot.data.joint_pos.torch[:, fingers].abs().max() < 0.02
            else:
                driver, _ = robot.find_joints(["finger_joint"])
                assert robot.data.joint_pos.torch[:, driver].min() > 0.7
        base, _ = robot.find_bodies(["base_link"])
        pose = robot.data.body_link_pose_w.torch[:, base[0]].clone()
        env.scene["probe"].write_root_pose_to_sim_index(root_pose=pose)
        env.sim.forward()
        backend = env.sim.backend_manager.backend
        contacts = backend.collision_pipeline.contacts()
        backend.collision_pipeline.collide(backend.state_0, contacts)
        count = int(contacts.contact_counters.numpy()[0])
        bodies = backend.model.shape_body.numpy()
        pairs = zip(
            bodies[contacts.rigid_contact_shape0.numpy()[:count]],
            bodies[contacts.rigid_contact_shape1.numpy()[:count]],
        )
        labels = backend.model.body_label
        assert any(
            (labels[a].endswith("/base_link") and "/probe/" in labels[b] + "/")
            or (labels[b].endswith("/base_link") and "/probe/" in labels[a] + "/")
            for a, b in pairs
            if a >= 0 and b >= 0
        )
        # Close on a fixed 40 mm object and measure overlap against the actual visual pad surfaces.
        # Free-space open/close alone misses the transient finger penetration caused by coarse contacts.
        stage = sim_utils.get_current_stage()
        xforms = UsdGeom.XformCache()
        pad_meshes, pad_centers = [], []
        for name in ("left_inner_finger", "right_inner_finger"):
            body = next(prim for prim in stage.Traverse() if prim.GetName() == name)
            visual = next(
                prim
                for prim in Usd.PrimRange(body)
                if prim.IsA(UsdGeom.Mesh) and "fingertipsstep" in prim.GetName() and "_hull_" not in prim.GetName()
            )
            mesh = get_mesh(visual, compute_inertia=False, apply_stage_units=False, load_visual_materials=False)
            transform = np.array(
                xforms.GetLocalToWorldTransform(visual) * xforms.GetLocalToWorldTransform(body).GetInverse()
            )
            vertices = mesh.vertices @ transform[:3, :3] + transform[3, :3]
            surface = trimesh.Trimesh(vertices, mesh.indices.reshape(-1, 3), process=True)
            pad_meshes.append(surface)
            pad_centers.append(surface.bounds.mean(axis=0))
        tips, _ = robot.find_bodies(["left_inner_finger", "right_inner_finger"], preserve_order=True)
        poses = robot.data.body_link_pose_w.torch[:, tips]
        centers = torch.tensor(np.array(pad_centers), device=device, dtype=poses.dtype)
        center = (quat_apply(poses[..., 3:], centers[None]) + poses[..., :3]).mean(dim=1)
        pose[:, :3] = center
        pose[:, 3:] = pose.new_tensor([0.0, 0.0, 0.0, 1.0])
        env.scene["probe"].write_root_pose_to_sim_index(root_pose=pose)
        action[:, -1] = -1
        for _ in range(30):
            env.step(action)
            poses = robot.data.body_link_pose_w.torch[:, tips]
            local_centers = quat_apply_inverse(poses[..., 3:], center[:, None] - poses[..., :3])[0].cpu().numpy()
            for surface, point in zip(pad_meshes, local_centers):
                distance = trimesh.proximity.closest_point(surface, point[None])[1][0]
                inside = surface.contains(point[None])[0]
                penetration = 0.02 + distance if inside else 0.02 - distance
                assert penetration < 0.003, f"Finger penetrated the probe by {penetration * 1000:.1f} mm"
        assert robot.data.joint_pos.torch[:, driver].max() < 0.7, "The probe must block full gripper closure"
    finally:
        env.close()
