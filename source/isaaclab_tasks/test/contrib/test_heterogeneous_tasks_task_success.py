# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical reset and task availability for bounded tabletop worlds, without downloaded assets."""

import itertools
from pathlib import Path

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    work_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/work_table",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.0, 0.05),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.175), rot=(0.0, 0.0, 0.14943813, 0.98877108)),
    )
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
    cube = UsdGeom.Cube.Define(stage, "/plate/collider")
    UsdGeom.Xformable(cube).AddScaleOp().Set((0.11, 0.1, 0.01))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    bodies = {
        "small": sim_utils.CuboidCfg(size=(0.04, 0.04, 0.04)),
        "large": sim_utils.CuboidCfg(size=(0.3, 0.25, 0.06)),
        "plate": sim_utils.UsdFileCfg(usd_path=str(plate_path)),
        "box": sim_utils.UsdFileCfg(usd_path=str(path)),
        "__empty__": sim_utils.SphereCfg(radius=0.001, visible=False),
    }
    worlds = (
        ("small", "plate", "__empty__"),
        ("small", "small", "__empty__"),
        ("large", "plate", "__empty__"),
        ("small", "box", "plate"),
    )
    cfg = _EnvCfg(seed=42)
    cfg.sim.device = request.param
    names = [f"object_{slot}" for slot in range(3)]
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
            if label in ("plate", "box"):
                spawn.semantic_tags.append(("category", "plate" if label == "plate" else "container"))
            candidates.append(spawn)
        setattr(
            cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=candidates, static=False, random_choice=False),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0, -10, 0.5)),
            ),
        )
    cfg.scene.clone_cfg.clone_combinations = [InclusionSet(assets=names, match_tags=["case"])]
    cfg.events = {
        "reset": EventTermCfg(func=isaaclab_mdp.reset_scene_to_default, mode="reset"),
        "place_objects": EventTermCfg(
            func=mdp.reset_tabletop,
            mode="reset",
            params={
                "objects": names,
                "table": SceneEntityCfg("work_table"),
                "workspace": ((0.25, -0.35, 0.1), (0.9, 0.35, 0.6)),
                "gripper_body": "gripper",
                "clearance": 0.005,
                "max_attempts": 256,
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
    container_world = next(i for i, names in enumerate(state.object_names) if "box" in names)
    large_world = next(i for i, names in enumerate(state.object_names) if "large" in names)
    assert [len(state.available_tasks[i]) for i in (0, 1, large_world, container_world)] == [1, 0, 0, 2]
    assert sorted(state.active.sum(1).tolist()) == [2, 2, 2, 3]
    assert not state.penetrating_worlds().any()
    assert state.instructions[0] and state.instructions[container_world]
    assert [state.instructions[i] for i in (1, large_world)] == ["", ""]
    with pytest.raises(ValueError, match="Unavailable"):
        state.set_task([1], [0])
    poses = torch.stack([asset.data.root_link_pose_w.torch for asset in state.assets], 1)
    before = poses.clone()
    for _ in range(3):
        env.reset(env_ids=torch.tensor([0, container_world], device=env.device))
        poses = torch.stack([asset.data.root_link_pose_w.torch for asset in state.assets], 1)
        torch.testing.assert_close(poses[[1, large_world]], before[[1, large_world]])
        assert not torch.equal(poses[0], before[0])
        corners = quat_apply(poses[..., None, 3:].expand(-1, -1, 8, 4), state.local_corners) + poses[..., None, :3]
        local = corners - env.scene.env_origins[:, None, None]
        assert (local[state.active] >= local.new_tensor([0.25, -0.35, 0.1])).all()
        assert (local[state.active] <= local.new_tensor([0.9, 0.35, 0.6])).all()
        table = state.table.data.root_link_pose_w.torch
        table_local = quat_apply_inverse(
            table[:, None, None, 3:].expand(-1, 3, 8, 4), corners - table[:, None, None, :3]
        )
        assert (table_local[..., :2][state.active] >= table_local.new_tensor([-0.45, -0.5]) + 0.005 - 1e-6).all()
        assert (table_local[..., :2][state.active] <= table_local.new_tensor([0.45, 0.5]) - 0.005 + 1e-6).all()
        assert not state.penetrating_worlds().any()
    assert not env.termination_manager.compute().any()
    inside = next(i for i, task in enumerate(state.available_tasks[container_world]) if task.family == "in_box")
    state.set_task([0, container_world], [0, inside])
    env.reset()
    target = torch.stack([asset.data.root_link_pose_w.torch for asset in state.assets], 1)
    for world, height in ((0, 0.032), (container_world, 0.031)):
        pose = target[world, state.target_ids[world]].clone()
        pose[:3] += quat_apply(pose[3:][None], pose.new_tensor([[0, 0, height]]))[0]
        ids = torch.tensor([world], device=env.device)
        state.assets[0].write_root_pose_to_sim_index(root_pose=pose[None], env_ids=ids)
        state.assets[0].write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids
        )
    for _ in range(60):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    expected = [False] * env.num_envs
    expected[0] = expected[container_world] = True
    assert env.termination_manager.compute().tolist() == expected
    params = dict(env.cfg.events["place_objects"].params, workspace=((2, 2, 0), (3, 3, 1)))
    with pytest.raises(RuntimeError, match="intersection"):
        state(env, torch.tensor([0], device=env.device), **params)
