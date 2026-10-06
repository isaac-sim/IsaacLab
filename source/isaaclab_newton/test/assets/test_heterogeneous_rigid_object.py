# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless real-Newton coverage of a rigid object whose colliders differ between environments.

Environments 0 and 1 hold a one-collider cube and environments 2 and 3 a two-collider body, cloned from one
multi-asset spawner. MuJoCo needs identical worlds, so the scene runs on FeatherPGS.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.envs.mdp.events import randomize_rigid_body_material
from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonManager
from newton_test_utils import newton_sim_cfg

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

NUM_ENVS = 4
SHAPE_COUNTS = (1, 1, 2, 2)
"""Colliders of the object in each environment, as the sequential clone strategy assigns the two variants."""


def _write_two_collider_body(path: Path) -> str:
    """Author a rigid body with two box colliders and return the file path."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    body = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(1.0)
    for name, y in (("Left", -0.1), ("Right", 0.1)):
        cube = UsdGeom.Cube.Define(stage, f"/Object/{name}")
        cube.CreateSizeAttr(0.1)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.0, y, 0.0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.SetDefaultPrim(body)
    stage.Save()
    return str(path)


@pytest.fixture
def scene(tmp_path: Path) -> Iterator[InteractiveScene]:
    """A four-environment CPU scene whose object has one collider in two environments and two in the others."""
    cube = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    scene_cfg = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=2.0)
    scene_cfg.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[cube, sim_utils.UsdFileCfg(usd_path=_write_two_collider_body(tmp_path / "two_colliders.usda"))]
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    sim_cfg = newton_sim_cfg("cpu", solver_cfg=FeatherPGSSolverCfg(pgs_mode="split"))
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        yield scene


def _object_shapes(obj: RigidObject, env_id: int) -> list[int]:
    """Model shape indices of the object in one environment."""
    model = NewtonManager.get_model()
    articulation = obj.root_view.articulation_ids.numpy()[env_id, 0]
    body = model.joint_child.numpy()[model.articulation_start.numpy()[articulation]]
    return model.body_shapes[int(body)]


def test_colliders_may_differ_between_environments(scene: InteractiveScene):
    """The object initializes, reads its state, and writes one environment's pose without touching the others."""
    obj: RigidObject = scene["object"]
    assert obj.num_instances == NUM_ENVS
    assert [len(_object_shapes(obj, env_id)) for env_id in range(NUM_ENVS)] == list(SHAPE_COUNTS)

    pose = obj.data.root_link_pose_w.torch.clone()
    pose[2, 2] += 0.5
    obj.write_root_pose_to_sim_index(root_pose=pose[[2]], env_ids=torch.tensor([2]))
    torch.testing.assert_close(obj.data.root_link_pose_w.torch[:, 2], torch.tensor([1.0, 1.0, 1.5, 1.0]))


def _material_term(scene: InteractiveScene, friction: float, restitution: float) -> tuple:
    """A Newton material term for the object and the parameters it is called with."""
    params = {
        "asset_cfg": SceneEntityCfg("object"),
        "static_friction_range": (friction, friction),
        "dynamic_friction_range": (friction, friction),
        "restitution_range": (restitution, restitution),
        "num_buckets": 1,
    }
    env = SimpleNamespace(
        num_envs=NUM_ENVS, device="cpu", scene=scene, sim=SimpleNamespace(physics_manager=NewtonManager)
    )
    term = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    return term, env, params


@pytest.mark.parametrize("env_ids", [[1, 2], [3], [2, 0], None])
def test_material_reaches_exactly_the_selected_environments(scene: InteractiveScene, env_ids: list[int] | None):
    """Every collider of each selected environment gets the sample; other environments keep their values."""
    obj: RigidObject = scene["object"]
    model = NewtonManager.get_model()
    startup, env, startup_params = _material_term(scene, 0.2, 0.1)
    reset, _, reset_params = _material_term(scene, 0.8, 0.6)
    startup(env, None, **startup_params)

    reset(env, None if env_ids is None else torch.tensor(env_ids), **reset_params)

    selected = range(NUM_ENVS) if env_ids is None else env_ids
    expected_mu = model.shape_material_mu.numpy().copy()
    expected_restitution = model.shape_material_restitution.numpy().copy()
    for env_id in range(NUM_ENVS):
        shapes = _object_shapes(obj, env_id)
        expected_mu[shapes] = 0.8 if env_id in selected else 0.2
        expected_restitution[shapes] = 0.6 if env_id in selected else 0.1
    np.testing.assert_allclose(model.shape_material_mu.numpy(), expected_mu)
    np.testing.assert_allclose(model.shape_material_restitution.numpy(), expected_restitution)
