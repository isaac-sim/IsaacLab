# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino physical validation for free-body state and inertial parameters."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import RigidObject

from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.managers import EventTermCfg
from isaaclab.sim import build_simulation_context
from isaaclab.test.physics.parameter_validation.fixtures import (
    FREE_BODY_COM,
    FREE_BODY_INERTIA,
    FREE_BODY_MASS,
    FREE_BODY_SIZE,
    build_free_body_usd,
    make_free_body_cfg,
)
from isaaclab.test.physics.parameter_validation.oracles import (
    PROFILE_FREE_DT,
    PhysicalCase,
    assert_physical_close,
    predict_angular_wrench_step,
    predict_linear_wrench_step,
    predict_semi_implicit_motion,
)

_GRAVITY = (1.2, -2.4, -4.8)
_INITIAL_POSITION = (0.1, -0.2, 1.0)
_INITIAL_LINEAR_VELOCITY = (0.3, -0.1, 0.2)
_INITIAL_ANGULAR_VELOCITY = (0.0, 0.0, 0.4)
_COM_OFFSET = (0.12, 0.0, 0.0)


def _case(parameter_id: str, authoring: str, *, atol: float = 2.0e-4) -> PhysicalCase:
    return PhysicalCase(
        parameter_id=parameter_id,
        backend="newton-kamino",
        authoring_path=authoring,
        profile="PROFILE-FREE",
        dt=PROFILE_FREE_DT,
        rtol=5.0e-3,
        atol=atol,
    )


def _spawn_cfg(mass: float = FREE_BODY_MASS) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=(FREE_BODY_SIZE, FREE_BODY_SIZE, FREE_BODY_SIZE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.0,
            angular_damping=0.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
    )


@contextmanager
def _free_body_scene(
    kamino,
    authoring: str,
    *,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = FREE_BODY_MASS,
    inertia: tuple[float, float, float] = FREE_BODY_INERTIA,
    principal_axes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    center_of_mass: tuple[float, float, float] = FREE_BODY_COM,
):
    sim_gravity = gravity if authoring == "cfg" else (0.0, 0.0, 0.0)
    with build_simulation_context(
        device="cuda:0",
        sim_cfg=kamino.profile_free_cfg(gravity=sim_gravity),
    ) as sim:
        sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform")
        if authoring == "usd":
            build_free_body_usd(
                position=position,
                orientation=orientation,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                mass=mass,
                inertia=inertia,
                principal_axes=principal_axes,
                center_of_mass=center_of_mass,
            )
            if gravity != (0.0, 0.0, 0.0):
                gravity_tensor = torch.tensor(gravity)
                magnitude = float(torch.linalg.vector_norm(gravity_tensor))
                scene = UsdPhysics.Scene.Get(sim.stage, sim.cfg.physics_prim_path)
                scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*(gravity_tensor / magnitude).tolist()))
                scene.CreateGravityMagnitudeAttr().Set(magnitude)
            cfg = make_free_body_cfg(
                position=position,
                orientation=orientation,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
        else:
            cfg = make_free_body_cfg(
                spawn=_spawn_cfg(mass),
                position=position,
                orientation=orientation,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
        body = RigidObject(cfg)
        sim.reset()
        body.write_root_link_pose_to_sim_index(root_pose=body.data.default_root_pose.torch)
        body.write_root_com_velocity_to_sim_index(root_velocity=body.data.default_root_vel.torch)
        body.update(0.0)
        yield sim, body


def _set_runtime_gravity(sim, gravity: tuple[float, float, float]) -> None:
    env = SimpleNamespace(
        device="cuda:0",
        num_envs=1,
        sim=sim,
        scene=SimpleNamespace(_ALL_INDICES=torch.tensor([0], device="cuda:0", dtype=torch.long)),
    )
    params = {
        "gravity_distribution_params": (list(gravity), list(gravity)),
        "operation": "abs",
        "distribution": "uniform",
    }
    cfg = EventTermCfg(func=randomize_physics_scene_gravity, mode="reset", params=params)
    term = randomize_physics_scene_gravity(cfg, env)
    term(env, None, **params)


@pytest.mark.parametrize("authoring", ["cfg", "runtime"])
def test_sim_01_gravity_vector(kamino, authoring):
    """SIM-01: Authored gravity produces the pinned discrete free-fall trajectory."""
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    scene_gravity = _GRAVITY if authoring != "runtime" else (0.0, 0.0, 0.0)
    with _free_body_scene(
        kamino,
        scene_authoring,
        gravity=scene_gravity,
        position=_INITIAL_POSITION,
    ) as (sim, body):
        if authoring == "runtime":
            sim.step()
            body.update(PROFILE_FREE_DT)
            pose = torch.tensor(
                [[*_INITIAL_POSITION, 0.0, 0.0, 0.0, 1.0]],
                device="cuda:0",
            )
            body.write_root_link_pose_to_sim_index(root_pose=pose)
            body.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device="cuda:0"))
            _set_runtime_gravity(sim, _GRAVITY)

        position_initial = torch.tensor(_INITIAL_POSITION, device="cuda:0")
        velocity_initial = torch.zeros(3, device="cuda:0")
        acceleration = torch.tensor(_GRAVITY, device="cuda:0")
        case = _case("SIM-01", authoring)
        for step in range(1, 4):
            sim.step()
            body.update(PROFILE_FREE_DT)
            position_expected, velocity_expected = predict_semi_implicit_motion(
                position_initial,
                velocity_initial,
                acceleration,
                dt=PROFILE_FREE_DT,
                steps=step,
            )
            assert_physical_close(body.data.root_link_pos_w.torch[0], position_expected, case)
            assert_physical_close(body.data.root_com_lin_vel_w.torch[0], velocity_expected, case)


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
def test_state_01_initial_and_live_link_pose(kamino, authoring):
    """STATE-01: Reset-default and live writes establish the requested link pose."""
    target_position = torch.tensor(_INITIAL_POSITION, device="cuda:0")
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    position = _INITIAL_POSITION if authoring != "runtime" else (0.0, 0.0, 0.0)
    with _free_body_scene(kamino, scene_authoring, position=position) as (sim, body):
        disturbance = torch.tensor([[0.7, 0.8, 0.9, 0.0, 0.0, 0.0, 1.0]], device="cuda:0")
        body.write_root_link_pose_to_sim_index(root_pose=disturbance)
        if authoring == "usd":
            sim.reset()
        elif authoring == "cfg":
            body.write_root_link_pose_to_sim_index(root_pose=body.data.default_root_pose.torch)
        else:
            target_pose = torch.tensor(
                [[*_INITIAL_POSITION, 0.0, 0.0, 0.0, 1.0]],
                device="cuda:0",
            )
            body.write_root_link_pose_to_sim_index(root_pose=target_pose)
        body.update(0.0)
        case = _case("STATE-01", authoring)
        assert_physical_close(body.data.root_link_pos_w.torch[0], target_position, case)
        sim.step()
        body.update(PROFILE_FREE_DT)
        assert_physical_close(body.data.root_link_pos_w.torch[0], target_position, case)


@pytest.mark.parametrize(
    "authoring",
    [
        pytest.param(
            "usd",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Kamino hard reset currently restores the floating-base joint state instead of USD velocity",
            ),
        ),
        "cfg",
        "runtime",
    ],
)
def test_state_02_initial_and_live_com_velocity(kamino, authoring):
    """STATE-02: Reset-default and live writes establish COM spatial velocity."""
    target_velocity = torch.tensor(
        [*_INITIAL_LINEAR_VELOCITY, *_INITIAL_ANGULAR_VELOCITY],
        device="cuda:0",
    )
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    linear_velocity = _INITIAL_LINEAR_VELOCITY if authoring != "runtime" else (0.0, 0.0, 0.0)
    angular_velocity = _INITIAL_ANGULAR_VELOCITY if authoring != "runtime" else (0.0, 0.0, 0.0)
    with _free_body_scene(
        kamino,
        scene_authoring,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    ) as (sim, body):
        body.write_root_com_velocity_to_sim_index(root_velocity=torch.full((1, 6), -0.3, device="cuda:0"))
        if authoring == "usd":
            sim.reset()
        elif authoring == "cfg":
            body.write_root_com_velocity_to_sim_index(root_velocity=body.data.default_root_vel.torch)
        else:
            body.write_root_com_velocity_to_sim_index(root_velocity=target_velocity.unsqueeze(0))
        body.update(0.0)
        case = _case("STATE-02", authoring)
        assert_physical_close(body.data.root_com_vel_w.torch[0], target_velocity, case)
        position_initial = body.data.root_com_pos_w.torch[0].clone()
        sim.step()
        body.update(PROFILE_FREE_DT)
        assert_physical_close(body.data.root_com_vel_w.torch[0], target_velocity, case)
        assert_physical_close(
            body.data.root_com_pos_w.torch[0],
            position_initial + target_velocity[:3] * PROFILE_FREE_DT,
            case,
        )


@pytest.mark.parametrize(
    "authoring",
    [
        "usd",
        "cfg",
        pytest.param(
            "runtime",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Kamino does not yet refresh inverse mass after the public runtime mass writer",
            ),
        ),
    ],
)
def test_body_01_mass_wrench_response(kamino, authoring):
    """BODY-01: Authored mass controls COM acceleration under a known force."""
    target_mass = 2.5
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    mass = target_mass if authoring != "runtime" else 1.0
    with _free_body_scene(kamino, scene_authoring, mass=mass) as (sim, body):
        if authoring == "runtime":
            body.set_masses_index(masses=torch.tensor([[target_mass]], device="cuda:0"))
        force = torch.tensor([[[4.0, -2.0, 1.0]]], device="cuda:0")
        body.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force,
            torques=torch.zeros_like(force),
            is_global=True,
        )
        body.write_data_to_sim()
        sim.step()
        body.update(PROFILE_FREE_DT)
        expected = predict_linear_wrench_step(
            torch.zeros(3, device="cuda:0"),
            force[0, 0],
            target_mass,
        )
        assert_physical_close(body.data.root_com_lin_vel_w.torch[0], expected, _case("BODY-01", authoring))


@pytest.mark.parametrize(
    "authoring",
    [
        "usd",
        pytest.param(
            "runtime",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Kamino does not yet refresh inverse inertia after the public runtime inertia writer",
            ),
        ),
    ],
)
@pytest.mark.parametrize("torque_axis", [0, 1])
def test_body_02_inertia_wrench_response(kamino, authoring, torque_axis):
    """BODY-02: Authored inertia controls angular acceleration under a known torque."""
    angle = torch.tensor(torch.pi / 4.0)
    principal_axes = (float(torch.cos(angle / 2.0)), 0.0, 0.0, float(torch.sin(angle / 2.0)))
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        device="cuda:0",
    )
    diagonal = torch.diag(torch.tensor(FREE_BODY_INERTIA, device="cuda:0"))
    inertia = rotation @ diagonal @ rotation.T
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    with _free_body_scene(
        kamino,
        scene_authoring,
        inertia=FREE_BODY_INERTIA,
        principal_axes=principal_axes,
    ) as (sim, body):
        if authoring == "runtime":
            body.set_inertias_index(inertias=inertia.reshape(1, 1, 9))
        torque = torch.zeros((1, 1, 3), device="cuda:0")
        torque[0, 0, torque_axis] = 0.6
        body.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.zeros_like(torque),
            torques=torque,
            is_global=True,
        )
        body.write_data_to_sim()
        sim.step()
        body.update(PROFILE_FREE_DT)
        expected = predict_angular_wrench_step(
            torch.zeros(3, device="cuda:0"),
            torque[0, 0],
            inertia,
        )
        assert_physical_close(
            body.data.root_com_ang_vel_w.torch[0],
            expected,
            _case("BODY-02", authoring, atol=5.0e-4),
        )


@pytest.mark.parametrize(
    "authoring",
    [
        "usd",
        "runtime",
    ],
)
def test_body_03_center_of_mass_force_response(kamino, authoring):
    """BODY-03: A force applied at the authored COM translates without rotation."""
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    com = _COM_OFFSET if authoring == "usd" else FREE_BODY_COM
    with _free_body_scene(kamino, scene_authoring, center_of_mass=com) as (sim, body):
        if authoring == "runtime":
            assert_physical_close(
                body.data.body_com_pos_w.torch[0, 0],
                torch.zeros(3, device="cuda:0"),
                _case("BODY-03", authoring),
            )
            body.set_coms_index(
                coms=wp.from_torch(
                    torch.tensor([[[_COM_OFFSET[0], _COM_OFFSET[1], _COM_OFFSET[2]]]], device="cuda:0"),
                    dtype=wp.vec3f,
                )
            )
        force = torch.tensor([[[0.0, 3.0, 0.0]]], device="cuda:0")
        position = torch.tensor([[list(_COM_OFFSET)]], device="cuda:0")
        body.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force,
            torques=torch.zeros_like(force),
            positions=position,
            is_global=True,
        )
        body.write_data_to_sim()
        sim.step()
        body.update(PROFILE_FREE_DT)
        expected_velocity = predict_linear_wrench_step(
            torch.zeros(3, device="cuda:0"),
            force[0, 0],
            FREE_BODY_MASS,
        )
        # Kamino's maximal-coordinate point-wrench solve couples translation and rotation at O(dt);
        # retain a bounded linear response while the no-rotation invariant is the COM oracle.
        assert_physical_close(
            body.data.root_com_lin_vel_w.torch[0],
            expected_velocity,
            _case("BODY-03", authoring, atol=2.5e-3),
        )
        assert_physical_close(
            body.data.root_com_ang_vel_w.torch[0],
            torch.zeros(3, device="cuda:0"),
            _case("BODY-03", authoring, atol=5.0e-4),
        )
