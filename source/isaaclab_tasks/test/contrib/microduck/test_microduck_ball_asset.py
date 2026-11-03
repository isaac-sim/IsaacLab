# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test the authored MicroDuck ball against an independent MuJoCo model.

Check the hollow-shell inertia (2/3) m r^2 and compliant ground contact, which geometry alone does not
define.
"""

import copy

import pytest
from isaaclab_newton.physics import NewtonCfg
from microduck_asset_test_utils import source_mjcf_path, spawn_test_assets

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_tasks.contrib.microduck.assets import (
    MICRODUCK_BALL_CFG,
    MICRODUCK_BALL_MASS,
    MICRODUCK_BALL_RADIUS,
    MICRODUCK_BALL_SLIDING_FRICTION,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

BALL_MJCF_FILENAME = "ball.xml"
"""Optional ball MJCF fixture filename."""

SOLID_SPHERE_INERTIA = 2.0 / 5.0 * MICRODUCK_BALL_MASS * MICRODUCK_BALL_RADIUS**2
"""Inertia [kg m^2] a uniform-density sphere of this mass and radius would have, 7.35e-6.

This is what USD and Newton resolve for a sphere prim carrying only a mass, and it is 40 % below the
MJCF's. It is spelled out so the shell assertion below fails against a *named* wrong answer rather
than merely against a number.
"""

EXPECTED_TORSIONAL_FRICTION = 0.005
EXPECTED_ROLLING_FRICTION = 0.0001
"""MuJoCo's default torsional and rolling friction, which the ball's MJCF leaves untouched.

Newton's shape defaults are the same two values, so the configuration authors neither. They are
pinned here anyway: a rolling ball is exactly the object whose trajectory the rolling coefficient
decides, and a backend default that drifted would change it silently.
"""


@pytest.fixture(scope="module")
def ball_mjcf_path() -> str:
    """Read the independently pinned ball reference supplied by the caller."""
    return source_mjcf_path(BALL_MJCF_FILENAME)


@pytest.fixture(scope="module")
def mj_ball(ball_mjcf_path):
    """Compile the optional ball MJCF as an independent physics reference."""
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_path(ball_mjcf_path)


@pytest.fixture(scope="module")
def newton_ball():
    """The configured ball spawned on a Newton stage, with the stage and the resolved model."""
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0", physics=NewtonCfg()))

    cfg = copy.deepcopy(MICRODUCK_BALL_CFG)
    cfg.prim_path = "/World/envs/env_0/Ball"
    ball = spawn_test_assets({"ball": cfg})["ball"]
    sim.reset()

    yield ball, sim.stage, sim.backend_manager.backend.model

    sim.stop()
    sim.clear_instance()


def _colliders(stage, prim_path: str) -> list[Usd.Prim]:
    """The spawned prims of the ball that take part in contact."""
    return [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]


def test_the_configuration_constants_are_the_mjcfs(mj_ball):
    """Mass, radius and sliding friction are read off the compiled MJCF, not chosen here."""
    import mujoco

    # MuJoCo body 0 is the world body; the model holds exactly one other
    assert mj_ball.nbody == 2
    assert mj_ball.ngeom == 1
    assert mujoco.mj_id2name(mj_ball, mujoco.mjtObj.mjOBJ_BODY, 1) == "ball"
    assert mj_ball.geom_type[0] == mujoco.mjtGeom.mjGEOM_SPHERE

    assert float(mj_ball.body_mass[1]) == pytest.approx(MICRODUCK_BALL_MASS, rel=1e-9)
    assert float(mj_ball.geom_size[0, 0]) == pytest.approx(MICRODUCK_BALL_RADIUS, rel=1e-9)
    sliding, torsional, rolling = (float(value) for value in mj_ball.geom_friction[0])
    assert sliding == pytest.approx(MICRODUCK_BALL_SLIDING_FRICTION, rel=1e-9)
    assert torsional == pytest.approx(EXPECTED_TORSIONAL_FRICTION, rel=1e-9)
    assert rolling == pytest.approx(EXPECTED_ROLLING_FRICTION, rel=1e-9)


def test_the_mjcf_inertia_is_a_hollow_shell(mj_ball):
    """The authored ball inertia is (2/3) m r^2, so a solid-sphere (2/5) m r^2 tensor is incorrect."""
    shell = 2.0 / 3.0 * MICRODUCK_BALL_MASS * MICRODUCK_BALL_RADIUS**2

    for axis in range(3):
        assert float(mj_ball.body_inertia[1, axis]) == pytest.approx(shell, rel=1e-6)
    assert shell == pytest.approx(1.225e-5, rel=1e-6)
    assert shell / SOLID_SPHERE_INERTIA == pytest.approx(5.0 / 3.0)


def test_the_spawned_ball_carries_the_mjcf_mass_and_hollow_shell_inertia(newton_ball, mj_ball):
    """What reaches the solver is the MJCF's mass and the MJCF's inertia, on all three axes."""
    ball, _, _ = newton_ball
    mass = ball.data.body_mass.torch[0, 0].item()
    # the inertia arrives as a flattened 3x3
    inertia = ball.data.body_inertia.torch[0, 0].reshape(3, 3).cpu()

    assert mass == pytest.approx(float(mj_ball.body_mass[1]), rel=1e-5)
    for axis in range(3):
        assert inertia[axis, axis].item() == pytest.approx(float(mj_ball.body_inertia[1, axis]), rel=1e-4)
        # a sphere has no product of inertia in any frame
        for other in range(3):
            if other != axis:
                assert inertia[axis, other].item() == pytest.approx(0.0, abs=1e-12)
    # the failure this test exists for: the tensor a sphere prim resolves to without the override
    assert inertia[0, 0].item() != pytest.approx(SOLID_SPHERE_INERTIA, rel=1e-3)


def test_the_spawned_ball_is_one_free_sphere_of_the_mjcfs_radius(newton_ball, mj_ball):
    """The ball is one free rigid body with one spherical collider and no actuated joints."""
    ball, stage, _ = newton_ball

    assert ball.num_bodies == 1
    assert not hasattr(ball, "num_joints")

    colliders = _colliders(stage, ball.cfg.prim_path)
    assert len(colliders) == 1
    sphere = UsdGeom.Sphere(colliders[0])
    assert sphere
    assert sphere.GetRadiusAttr().Get() == pytest.approx(float(mj_ball.geom_size[0, 0]), rel=1e-9)
    # the root prim is the rigid body, and it is the one carrying the mass the inertia is derived on
    root = stage.GetPrimAtPath(ball.cfg.prim_path)
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root.HasAPI(UsdPhysics.MassAPI)


def test_the_collider_resolves_to_a_physics_material_carrying_the_mjcf_friction(newton_ball, mj_ball):
    """The friction is *bound* to the collider, not merely configured.

    This is pitfall 3 of the MicroDuck conversion checklist, restated for an authored asset: on the
    robots the importer wrote bindings that USD then dropped, and the feet silently slid on a backend
    default. A prop reaches the solver through a different path, so the binding is re-checked here
    both in USD and in the resolved Newton shape material.
    """
    ball, stage, model = newton_ball
    sliding, torsional, rolling = (float(value) for value in mj_ball.geom_friction[0])

    material, _ = UsdShade.MaterialBindingAPI(_colliders(stage, ball.cfg.prim_path)[0]).ComputeBoundMaterial("physics")
    assert material, "no physics material bound to the ball collider"
    material_api = UsdPhysics.MaterialAPI(material.GetPrim())
    # MuJoCo has one sliding coefficient where UsdPhysics has a static and a dynamic one
    assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(sliding, rel=1e-6)
    assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(sliding, rel=1e-6)
    # MuJoCo has no restitution: its bounce comes out of the contact solver reference
    assert material_api.GetRestitutionAttr().Get() == pytest.approx(0.0)

    # and what the solver actually holds, which is the only reading a contact uses
    assert len(model.shape_material_mu.numpy()) == 1, "the stage carries shapes other than the ball"
    assert float(model.shape_material_mu.numpy()[0]) == pytest.approx(sliding, rel=1e-6)
    assert float(model.shape_material_mu_torsional.numpy()[0]) == pytest.approx(torsional, rel=1e-6)
    assert float(model.shape_material_mu_rolling.numpy()[0]) == pytest.approx(rolling, rel=1e-6)


def test_the_ball_spawns_resting_on_the_ground_at_the_mjcf_pose(newton_ball, mj_ball):
    """The pre-reset pose is the MJCF's, and it sets the ball down rather than dropping it."""
    ball, _, _ = newton_ball

    expected = [float(value) for value in mj_ball.body_pos[1]]
    assert list(MICRODUCK_BALL_CFG.init_state.pos) == pytest.approx(expected)
    # centre one radius up: resting on the plane, with no penetration and nothing to fall
    assert MICRODUCK_BALL_CFG.init_state.pos[2] == pytest.approx(MICRODUCK_BALL_RADIUS)
    assert ball.data.root_link_pos_w.torch[0, 2].item() == pytest.approx(MICRODUCK_BALL_RADIUS, abs=1e-6)


def test_the_spawner_refuses_a_configuration_it_cannot_derive_an_inertia_from():
    """The hollow-shell tensor is derived from the mass, so a configuration without one is an error.

    Silently falling back to the solid-sphere tensor is the failure mode this whole file is about,
    so it is refused rather than defaulted.
    """
    spawn = copy.deepcopy(MICRODUCK_BALL_CFG.spawn)
    spawn.mass_props = None

    with pytest.raises(ValueError, match="mass_props.mass"):
        spawn.func("/World/BallWithoutMass", spawn)
