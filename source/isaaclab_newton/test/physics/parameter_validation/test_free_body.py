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
_INITIAL_ORIENTATION = (0.0, 0.0, 0.38268343, 0.92387953)
_INITIAL_LINEAR_VELOCITY = (0.3, -0.1, 0.2)
_INITIAL_ANGULAR_VELOCITY = (0.0, 0.0, 0.4)
_COM_OFFSET = (0.12, 0.0, 0.0)
_CFG_SENTINEL_POSITION = (-0.6, 0.7, 1.3)
_CFG_SENTINEL_LINEAR_VELOCITY = (-0.5, 0.6, -0.7)
_CFG_SENTINEL_ANGULAR_VELOCITY = (0.8, -0.9, 1.0)

_PUBLIC_APIS = {
    "SIM-01": "SimulationCfg.gravity or randomize_physics_scene_gravity",
    "STATE-01": "write_root_link_pose_to_sim_index",
    "STATE-02": "write_root_com_velocity_to_sim_index",
    "BODY-01": "set_masses_index",
    "BODY-02": "set_inertias_index",
    "BODY-03": "set_coms_index",
}


def _case(parameter_adapter, parameter_id: str, authoring: str, *, atol: float = 2.0e-4) -> PhysicalCase:
    api = _PUBLIC_APIS[parameter_id]
    if authoring == "usd":
        api = "UsdPhysics RigidBodyAPI or MassAPI"
    elif authoring == "cfg":
        api = "RigidObjectCfg initialization"
    return PhysicalCase(
        parameter_id=parameter_id,
        backend=parameter_adapter.backend,
        authoring_path=authoring,
        profile="PROFILE-FREE",
        dt=PROFILE_FREE_DT,
        substeps=1,
        api=api,
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
    parameter_adapter,
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
    cfg_position: tuple[float, float, float] | None = None,
    cfg_orientation: tuple[float, float, float, float] | None = None,
    cfg_linear_velocity: tuple[float, float, float] | None = None,
    cfg_angular_velocity: tuple[float, float, float] | None = None,
):
    sim_gravity = gravity if authoring == "cfg" else (0.0, 0.0, 0.0)
    with build_simulation_context(
        device="cuda:0",
        sim_cfg=parameter_adapter.profile_free_cfg(gravity=sim_gravity),
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
                position=position if cfg_position is None else cfg_position,
                orientation=orientation if cfg_orientation is None else cfg_orientation,
                linear_velocity=linear_velocity if cfg_linear_velocity is None else cfg_linear_velocity,
                angular_velocity=angular_velocity if cfg_angular_velocity is None else cfg_angular_velocity,
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
        if authoring != "usd":
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
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_sim_01_gravity_vector(parameter_adapter, authoring):
    """SIM-01: Authored gravity produces the pinned discrete free-fall trajectory."""
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    scene_gravity = _GRAVITY if authoring != "runtime" else (0.0, 0.0, 0.0)
    with _free_body_scene(
        parameter_adapter,
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
        case = _case(parameter_adapter, "SIM-01", authoring)
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
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_state_01_initial_and_live_link_pose(parameter_adapter, authoring):
    """STATE-01: Reset-default and live writes establish the requested link pose."""
    target_position = torch.tensor(_INITIAL_POSITION, device="cuda:0")
    target_orientation = torch.tensor(_INITIAL_ORIENTATION, device="cuda:0")
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    position = _INITIAL_POSITION if authoring != "runtime" else (0.0, 0.0, 0.0)
    orientation = _INITIAL_ORIENTATION if authoring != "runtime" else (0.0, 0.0, 0.0, 1.0)
    with _free_body_scene(
        parameter_adapter,
        scene_authoring,
        position=position,
        orientation=orientation,
        cfg_position=_CFG_SENTINEL_POSITION if authoring == "usd" else None,
    ) as (sim, body):
        disturbance = torch.tensor([[0.7, 0.8, 0.9, 0.0, 0.0, 0.0, 1.0]], device="cuda:0")
        body.write_root_link_pose_to_sim_index(root_pose=disturbance)
        if authoring == "usd":
            assert_physical_close(
                body.data.default_root_pose.torch[0, :3],
                torch.tensor(_CFG_SENTINEL_POSITION, device="cuda:0"),
                _case(parameter_adapter, "STATE-01", authoring),
            )
            sim.reset()
        elif authoring == "cfg":
            body.write_root_link_pose_to_sim_index(root_pose=body.data.default_root_pose.torch)
        else:
            target_pose = torch.tensor(
                [[*_INITIAL_POSITION, *_INITIAL_ORIENTATION]],
                device="cuda:0",
            )
            body.write_root_link_pose_to_sim_index(root_pose=target_pose)
        body.update(0.0)
        case = _case(parameter_adapter, "STATE-01", authoring)
        assert_physical_close(body.data.root_link_pos_w.torch[0], target_position, case)
        assert_physical_close(body.data.root_link_quat_w.torch[0], target_orientation, case)
        sim.step()
        body.update(PROFILE_FREE_DT)
        assert_physical_close(body.data.root_link_pos_w.torch[0], target_position, case)
        assert_physical_close(body.data.root_link_quat_w.torch[0], target_orientation, case)


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_state_02_initial_and_live_com_velocity(parameter_adapter, authoring):
    """STATE-02: Reset-default and live writes establish COM spatial velocity."""
    target_velocity = torch.tensor(
        [*_INITIAL_LINEAR_VELOCITY, *_INITIAL_ANGULAR_VELOCITY],
        device="cuda:0",
    )
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    linear_velocity = _INITIAL_LINEAR_VELOCITY if authoring != "runtime" else (0.0, 0.0, 0.0)
    angular_velocity = _INITIAL_ANGULAR_VELOCITY if authoring != "runtime" else (0.0, 0.0, 0.0)
    with _free_body_scene(
        parameter_adapter,
        scene_authoring,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        cfg_linear_velocity=_CFG_SENTINEL_LINEAR_VELOCITY if authoring == "usd" else None,
        cfg_angular_velocity=_CFG_SENTINEL_ANGULAR_VELOCITY if authoring == "usd" else None,
    ) as (sim, body):
        body.write_root_com_velocity_to_sim_index(root_velocity=torch.full((1, 6), -0.3, device="cuda:0"))
        if authoring == "usd":
            assert_physical_close(
                body.data.default_root_vel.torch[0],
                torch.tensor(
                    [*_CFG_SENTINEL_LINEAR_VELOCITY, *_CFG_SENTINEL_ANGULAR_VELOCITY],
                    device="cuda:0",
                ),
                _case(parameter_adapter, "STATE-02", authoring),
            )
            sim.reset()
        elif authoring == "cfg":
            body.write_root_com_velocity_to_sim_index(root_velocity=body.data.default_root_vel.torch)
        else:
            body.write_root_com_velocity_to_sim_index(root_velocity=target_velocity.unsqueeze(0))
        body.update(0.0)
        case = _case(parameter_adapter, "STATE-02", authoring)
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


_BODY_01_MASS_XFAIL_REASON = (
    "IsaacLab#6518: The articulation / rigid object does not refresh inverse mass after the public runtime mass writer"
)

_BODY_01_CASES = [
    pytest.param("kamino", "usd", id="kamino-usd"),
    pytest.param("kamino", "cfg", id="kamino-cfg"),
    pytest.param(
        "kamino",
        "runtime",
        id="kamino-runtime",
        marks=pytest.mark.xfail(strict=True, reason=_BODY_01_MASS_XFAIL_REASON),
    ),
    pytest.param("mjwarp", "usd", id="mjwarp-usd"),
    pytest.param("mjwarp", "cfg", id="mjwarp-cfg"),
    pytest.param("mjwarp", "runtime", id="mjwarp-runtime"),
]


@pytest.mark.parametrize(("parameter_adapter", "authoring"), _BODY_01_CASES, indirect=["parameter_adapter"])
def test_body_01_mass_wrench_response(parameter_adapter, authoring):
    """BODY-01: Authored mass controls COM acceleration under a known force."""
    target_mass = 2.5
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    mass = target_mass if authoring != "runtime" else 1.0
    with _free_body_scene(parameter_adapter, scene_authoring, mass=mass) as (sim, body):
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
        assert_physical_close(
            body.data.root_com_lin_vel_w.torch[0], expected, _case(parameter_adapter, "BODY-01", authoring)
        )


_BODY_02_CASES = [
    pytest.param("kamino", "usd", id="kamino-usd"),
    pytest.param(
        "kamino",
        "runtime",
        id="kamino-runtime",
        marks=pytest.mark.xfail(strict=True, reason=_BODY_01_MASS_XFAIL_REASON),
    ),
    pytest.param("mjwarp", "usd", id="mjwarp-usd"),
    pytest.param("mjwarp", "runtime", id="mjwarp-runtime"),
]


@pytest.mark.parametrize(("parameter_adapter", "authoring"), _BODY_02_CASES, indirect=["parameter_adapter"])
@pytest.mark.parametrize("torque_axis", [0, 1])
def test_body_02_inertia_wrench_response(parameter_adapter, authoring, torque_axis):
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
        parameter_adapter,
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
            _case(parameter_adapter, "BODY-02", authoring, atol=5.0e-4),
        )


def _run_com_force_response(
    parameter_adapter,
    authoring: str,
    center_of_mass: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a point force and return the resulting COM linear and angular velocities."""
    scene_authoring = authoring if authoring != "runtime" else "cfg"
    com = center_of_mass if authoring == "usd" else FREE_BODY_COM
    with _free_body_scene(parameter_adapter, scene_authoring, center_of_mass=com) as (sim, body):
        if authoring == "runtime":
            assert_physical_close(
                body.data.body_com_pos_w.torch[0, 0],
                torch.zeros(3, device="cuda:0"),
                _case(parameter_adapter, "BODY-03", authoring),
            )
            body.set_coms_index(
                coms=wp.from_torch(
                    torch.tensor([[[*center_of_mass]]], device="cuda:0"),
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
        return (
            body.data.root_com_lin_vel_w.torch[0].clone(),
            body.data.root_com_ang_vel_w.torch[0].clone(),
        )


@pytest.mark.parametrize("authoring", ["usd", "runtime"])
@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_body_03_center_of_mass_force_response(parameter_adapter, authoring):
    """BODY-03: A force at the authored COM translates without rotation."""
    linear_velocity, angular_velocity = _run_com_force_response(parameter_adapter, authoring, _COM_OFFSET)
    expected_velocity = predict_linear_wrench_step(
        torch.zeros(3, device="cuda:0"),
        torch.tensor([0.0, 3.0, 0.0], device="cuda:0"),
        FREE_BODY_MASS,
    )
    case = _case(parameter_adapter, "BODY-03", authoring, atol=2.5e-3)
    assert_physical_close(linear_velocity, expected_velocity, case)
    assert_physical_close(
        angular_velocity,
        torch.zeros(3, device="cuda:0"),
        _case(parameter_adapter, "BODY-03", authoring, atol=5.0e-4),
    )

    _, control_angular_velocity = _run_com_force_response(parameter_adapter, authoring, FREE_BODY_COM)
    assert torch.linalg.vector_norm(control_angular_velocity) > 10.0 * 5.0e-4, (
        "BODY-03: zero-COM control did not rotate under the same off-center force"
    )
