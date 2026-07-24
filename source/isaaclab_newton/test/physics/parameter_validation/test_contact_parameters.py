# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physical validation for contact material, geometry, and offset parameters."""

# pyright: reportAttributeAccessIssue=none, reportPrivateUsage=none

import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.sensors.contact_sensor import ContactSensor, ContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import (
    randomize_rigid_body_collider_offsets,
    randomize_rigid_body_material,
    randomize_rigid_body_scale,
)
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim import build_simulation_context
from isaaclab.test.physics.parameter_validation.fixtures import (
    _MIN_MU,
    CONTACT_BOX_SIZE,
    CONTACT_GROUND_PRIM_PATH,
    CONTACT_OBJECT_PRIM_PATH,
    CONTACT_SPHERE_RADIUS,
    build_contact_box_usd,
    build_contact_ground_usd,
    build_contact_sphere_usd,
    make_contact_box_cfg,
    make_contact_sphere_cfg,
    spawn_contact_ground,
)
from isaaclab.test.physics.parameter_validation.oracles import (
    PROFILE_CONTACT_DT,
    PhysicalCase,
    assert_physical_close,
    contact_separation_atol,
    dynamic_friction_distance_atol,
    predict_critical_incline_angle,
    predict_friction_stopping_distance,
    predict_rebound_height,
    static_friction_angle_deadband,
)

_DEVICE = "cuda:0"
_GRAVITY = 9.81
_GROUND_SIZE = (4.0, 4.0, 0.1)
_GROUND_TOP = 0.0
_CONTACT_FORCE_THRESHOLD = 0.1
_KAMINO_RESTITUTION_REASON = (
    "Accepted gap (Newton/Kamino integration): the selected Newton CollisionPipeline produces inelastic contacts"
)
_MJWARP_RESTITUTION_REASON = (
    "Accepted gap (Newton-MJWarp integration): the default MuJoCo-contact path does not turn the public "
    "Newton restitution value into a rebound"
)
_CONTACT_BACKENDS = [pytest.param("kamino", id="kamino"), pytest.param("mjwarp", id="mjwarp")]
_RESTITUTION_BACKENDS = [
    pytest.param(
        "kamino",
        id="kamino",
        marks=pytest.mark.xfail(strict=True, reason=_KAMINO_RESTITUTION_REASON),
    ),
    pytest.param(
        "mjwarp",
        id="mjwarp",
        marks=pytest.mark.xfail(strict=True, reason=_MJWARP_RESTITUTION_REASON),
    ),
]

_PUBLIC_APIS = {
    "MAT-03": "UsdPhysics material, spawn physics_material, or randomize_rigid_body_material",
    "MAT-04": "UsdPhysics material, spawn physics_material, or randomize_rigid_body_material",
    "SHAPE-01": "USD local transform or Isaac Lab spawn transform",
    "SHAPE-02": "USD dimensions, shape spawn config, or randomize_rigid_body_scale",
    "SHAPE-03": "USD radius or SphereCfg.radius",
    "CONTACT-01": "CollisionPropertiesCfg or randomize_rigid_body_collider_offsets",
}


class _AssetScene:
    """Minimal scene contract used by public event terms."""

    def __init__(self, assets: dict[str, RigidObject]):
        self._assets = assets
        self.num_envs = 1

    def __getitem__(self, name: str) -> RigidObject:
        return self._assets[name]


def _case(
    parameter_adapter,
    parameter_id: str,
    authoring: str,
    *,
    rtol: float,
    atol: float,
) -> PhysicalCase:
    return PhysicalCase(
        parameter_id=parameter_id,
        backend=parameter_adapter.backend,
        authoring_path=authoring,
        profile="PROFILE-CONTACT",
        dt=PROFILE_CONTACT_DT,
        substeps=parameter_adapter.contact_substeps,
        api=_PUBLIC_APIS[parameter_id],
        rtol=rtol,
        atol=atol,
    )


def _ramp_orientation(angle: float) -> tuple[float, float, float, float]:
    """Return a scalar-first quaternion for a ramp rotated around world Y."""
    return (math.cos(angle / 2.0), 0.0, -math.sin(angle / 2.0), 0.0)


@contextmanager
def _contact_scene(
    parameter_adapter,
    authoring: str,
    *,
    object_shape: str,
    object_position: tuple[float, float, float],
    object_size: tuple[float, float, float] = CONTACT_BOX_SIZE,
    object_radius: float = CONTACT_SPHERE_RADIUS,
    ground_angle: float = 0.0,
    mu: float = 0.0,
    restitution: float = 0.0,
    rest_offset: float = 0.0,
    contact_offset: float = 0.01,
    object_local_offset: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -_GRAVITY),
):
    """Build one controlled static-ground contact scene."""
    initial_mu = _MIN_MU if authoring == "runtime" else mu
    initial_restitution = 0.0 if authoring == "runtime" else restitution
    initial_rest_offset = 0.0 if authoring == "runtime" else rest_offset
    initial_contact_offset = 0.01 if authoring == "runtime" else contact_offset
    # Runtime event terms require a rigid asset, so author the static ground with its target properties and apply
    # runtime changes only to the dynamic object.
    ground_kwargs = {
        "size": _GROUND_SIZE,
        "position": (0.0, 0.0, -0.05),
        "orientation": _ramp_orientation(ground_angle),
        "mu": mu,
        "restitution": restitution,
        "rest_offset": rest_offset,
        "contact_offset": contact_offset,
    }
    object_box_kwargs = {
        "size": object_size,
        "position": object_position,
        "mu": initial_mu,
        "restitution": initial_restitution,
        "rest_offset": initial_rest_offset,
        "contact_offset": initial_contact_offset,
    }
    object_sphere_kwargs = {
        "radius": object_radius,
        "position": object_position,
        "mu": initial_mu,
        "restitution": initial_restitution,
        "rest_offset": initial_rest_offset,
        "contact_offset": initial_contact_offset,
    }

    with build_simulation_context(
        device=_DEVICE,
        sim_cfg=parameter_adapter.profile_contact_cfg(gravity=gravity),
    ) as sim:
        sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform")
        if authoring == "usd":
            build_contact_ground_usd(CONTACT_GROUND_PRIM_PATH, **ground_kwargs)
            if object_shape == "box":
                build_contact_box_usd(CONTACT_OBJECT_PRIM_PATH, **object_box_kwargs)
            else:
                build_contact_sphere_usd(CONTACT_OBJECT_PRIM_PATH, **object_sphere_kwargs)
            if object_shape == "box":
                object_cfg = make_contact_box_cfg(CONTACT_OBJECT_PRIM_PATH, spawn=False, **object_box_kwargs)
            else:
                object_cfg = make_contact_sphere_cfg(CONTACT_OBJECT_PRIM_PATH, spawn=False, **object_sphere_kwargs)
        else:
            spawn_contact_ground(CONTACT_GROUND_PRIM_PATH, **ground_kwargs)
            if object_shape == "box":
                object_cfg = make_contact_box_cfg(CONTACT_OBJECT_PRIM_PATH, spawn=True, **object_box_kwargs)
            elif object_shape == "sphere":
                object_cfg = make_contact_sphere_cfg(CONTACT_OBJECT_PRIM_PATH, spawn=True, **object_sphere_kwargs)
            else:
                raise ValueError(f"Unsupported contact object shape: {object_shape}")
        if object_local_offset != 0.0:
            if object_cfg.spawn is not None:
                if object_shape == "box":
                    build_contact_box_usd(CONTACT_OBJECT_PRIM_PATH, **object_box_kwargs)
                else:
                    build_contact_sphere_usd(CONTACT_OBJECT_PRIM_PATH, **object_sphere_kwargs)
                if object_shape == "box":
                    object_cfg = make_contact_box_cfg(CONTACT_OBJECT_PRIM_PATH, spawn=False, **object_box_kwargs)
                else:
                    object_cfg = make_contact_sphere_cfg(CONTACT_OBJECT_PRIM_PATH, spawn=False, **object_sphere_kwargs)
            mesh_prim = sim.stage.GetPrimAtPath(f"{CONTACT_OBJECT_PRIM_PATH}/geometry/mesh")
            mesh_prim.GetAttribute("xformOp:translate").Set((0.0, 0.0, -object_local_offset))
        body = RigidObject(object_cfg)
        sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=CONTACT_OBJECT_PRIM_PATH,
                update_period=0.0,
                history_length=1,
            )
        )
        sim.reset()
        body.update(0.0)
        sensor.update(0.0, force_recompute=True)

        env = SimpleNamespace(
            device=_DEVICE,
            num_envs=1,
            sim=sim,
            scene=_AssetScene({"object": body}),
        )
        if authoring == "runtime":
            sim.step()
            body.update(PROFILE_CONTACT_DT)
            body.write_root_link_pose_to_sim_index(root_pose=body.data.default_root_pose.torch)
            body.write_root_com_velocity_to_sim_index(root_velocity=body.data.default_root_vel.torch)
            _apply_runtime_material(env, "object", mu, restitution)
            if rest_offset != 0.0 or contact_offset != 0.0:
                _apply_runtime_offsets(env, "object", rest_offset, contact_offset)
        yield sim, body, sensor, env


def _apply_runtime_material(env, asset_name: str, mu: float, restitution: float) -> None:
    params = {
        "static_friction_range": (mu, mu),
        "dynamic_friction_range": (mu, mu),
        "restitution_range": (restitution, restitution),
        "num_buckets": 1,
        "asset_cfg": SceneEntityCfg(asset_name),
    }
    cfg = EventTermCfg(
        func=randomize_rigid_body_material,  # pyright: ignore[reportArgumentType]
        mode="reset",
        params=params,
    )
    term = randomize_rigid_body_material(cfg, env)
    term(env, None, **params)


def _apply_runtime_offsets(env, asset_name: str, rest_offset: float, contact_offset: float) -> None:
    params = {
        "asset_cfg": SceneEntityCfg(asset_name),
        "rest_offset_distribution_params": (rest_offset, rest_offset),
        "contact_offset_distribution_params": (contact_offset, contact_offset),
        "distribution": "uniform",
    }
    cfg = EventTermCfg(
        func=randomize_rigid_body_collider_offsets,  # pyright: ignore[reportArgumentType]
        mode="reset",
        params=params,
    )
    term = randomize_rigid_body_collider_offsets(cfg, env)
    term(env, None, **params)


def _step(sim, body: RigidObject, sensor: ContactSensor) -> float:
    sim.step()
    body.update(PROFILE_CONTACT_DT)
    sensor.update(PROFILE_CONTACT_DT, force_recompute=True)
    return float(torch.linalg.vector_norm(sensor.data.net_forces_w.torch[0]).item())


def _measure_ramp(parameter_adapter, authoring: str, angle: float, mu: float) -> tuple[float, float]:
    with _contact_scene(
        parameter_adapter,
        authoring,
        object_shape="box",
        object_position=(0.0, 0.0, 0.18),
        ground_angle=angle,
        mu=mu,
    ) as (sim, body, sensor, _):
        initial_position = body.data.root_link_pos_w.torch[0].clone()
        for _ in range(240):
            _step(sim, body, sensor)
        displacement = torch.linalg.vector_norm(body.data.root_link_pos_w.torch[0] - initial_position)
        speed = torch.linalg.vector_norm(body.data.root_com_lin_vel_w.torch[0])
        return float(displacement), float(speed)


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", _CONTACT_BACKENDS, indirect=True)
def test_mat_03_static_friction_incline_threshold(parameter_adapter, authoring):
    """MAT-03/FIX-FRICTION-STATIC: combined mu controls the inclined-plane threshold."""
    mu = 0.5
    critical_angle = predict_critical_incline_angle(mu)
    deadband = static_friction_angle_deadband(critical_angle)
    rest_displacement, rest_speed = _measure_ramp(parameter_adapter, authoring, critical_angle - deadband, mu)
    slide_displacement, _ = _measure_ramp(parameter_adapter, authoring, critical_angle + deadband, mu)
    case = _case(parameter_adapter, "MAT-03", authoring, rtol=0.0, atol=0.03)
    assert rest_speed <= 0.1, case.message((rest_displacement, rest_speed), "speed <= 0.1")
    assert slide_displacement >= 0.05, case.message(slide_displacement, "displacement >= 0.05")


def _measure_stopping_distance(parameter_adapter, authoring: str, mu: float, initial_speed: float) -> float:
    with _contact_scene(
        parameter_adapter,
        authoring,
        object_shape="box",
        object_position=(0.0, 0.0, 0.11),
        mu=mu,
    ) as (sim, body, sensor, _):
        for _ in range(60):
            _step(sim, body, sensor)
        velocity = torch.zeros((1, 6), device=_DEVICE)
        velocity[0, 0] = initial_speed
        body.write_root_com_velocity_to_sim_index(root_velocity=velocity)
        start = float(body.data.root_link_pos_w.torch[0, 0])
        for _ in range(480):
            _step(sim, body, sensor)
            if abs(float(body.data.root_com_lin_vel_w.torch[0, 0])) < 0.05:
                break
        return abs(float(body.data.root_link_pos_w.torch[0, 0]) - start)


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", _CONTACT_BACKENDS, indirect=True)
def test_mat_03_dynamic_friction_stopping_distance(parameter_adapter, authoring):
    """MAT-03/FIX-FRICTION-DYNAMIC: combined mu controls level-plane stopping distance."""
    mu = 0.5
    initial_speed = 2.0
    measured = _measure_stopping_distance(parameter_adapter, authoring, mu, initial_speed)
    expected = predict_friction_stopping_distance(initial_speed, mu, _GRAVITY)
    case = _case(
        parameter_adapter,
        "MAT-03",
        authoring,
        rtol=0.1,
        atol=dynamic_friction_distance_atol(initial_speed),
    )
    assert_physical_close(measured, expected, case)


def _measure_rebound_height(parameter_adapter, authoring: str, restitution: float) -> float:
    drop_height = 1.0
    with _contact_scene(
        parameter_adapter,
        authoring,
        object_shape="sphere",
        object_position=(0.0, 0.0, _GROUND_TOP + CONTACT_SPHERE_RADIUS + drop_height),
        mu=0.0,
        restitution=restitution,
    ) as (sim, body, sensor, _):
        impact_seen = False
        apex = _GROUND_TOP + CONTACT_SPHERE_RADIUS
        for _ in range(600):
            force = _step(sim, body, sensor)
            velocity_z = float(body.data.root_com_lin_vel_w.torch[0, 2])
            if force > _CONTACT_FORCE_THRESHOLD:
                impact_seen = True
            if impact_seen:
                apex = max(apex, float(body.data.root_com_pos_w.torch[0, 2]))
                if velocity_z < 0.0 and apex > _GROUND_TOP + CONTACT_SPHERE_RADIUS + 0.01:
                    break
        assert impact_seen, "MAT-04: sphere never produced a contact observation"
        return apex - _GROUND_TOP - CONTACT_SPHERE_RADIUS


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize("parameter_adapter", _RESTITUTION_BACKENDS, indirect=True)
def test_mat_04_restitution_first_rebound(parameter_adapter, authoring):
    """MAT-04/FIX-RESTITUTION: restitution controls the first post-impact rebound."""
    restitution = 0.8
    measured = _measure_rebound_height(parameter_adapter, authoring, restitution)
    case = _case(parameter_adapter, "MAT-04", authoring, rtol=0.03, atol=0.02)
    if parameter_adapter.backend == "newton-kamino":
        assert_physical_close(measured, predict_rebound_height(1.0, restitution), case)
    else:
        control = _measure_rebound_height(parameter_adapter, authoring, 0.0)
        assert measured > control + 0.1, case.message(measured, f"greater than inelastic control {control} by 0.1")


def _measure_first_contact_step(
    parameter_adapter,
    authoring: str,
    *,
    shape: str,
    size: tuple[float, float, float] = CONTACT_BOX_SIZE,
    radius: float = CONTACT_SPHERE_RADIUS,
    local_offset: float = 0.0,
) -> int:
    with _contact_scene(
        parameter_adapter,
        authoring,
        object_shape=shape,
        object_position=(0.0, 0.0, 0.8),
        object_size=size,
        object_radius=radius,
        object_local_offset=local_offset,
    ) as (sim, body, sensor, _):
        for step in range(180):
            if _step(sim, body, sensor) > _CONTACT_FORCE_THRESHOLD:
                return step
    raise AssertionError("Contact was not detected within the shape-contact probe")


@pytest.mark.parametrize("authoring", ["usd", "cfg"])
@pytest.mark.parametrize("parameter_adapter", _CONTACT_BACKENDS, indirect=True)
def test_shape_01_local_transform_changes_contact_time(parameter_adapter, authoring):
    """SHAPE-01/FIX-SHAPE-CONTACT: local collider transform changes first-contact time."""
    baseline = _measure_first_contact_step(parameter_adapter, authoring, shape="box")
    changed = _measure_first_contact_step(parameter_adapter, authoring, shape="box", local_offset=0.1)
    case = _case(parameter_adapter, "SHAPE-01", authoring, rtol=0.0, atol=1.0)
    assert changed < baseline, case.message(changed, f"earlier than baseline step {baseline}")


@pytest.mark.parametrize("authoring", ["usd", "cfg"])
@pytest.mark.parametrize("parameter_adapter", _CONTACT_BACKENDS, indirect=True)
def test_shape_02_dimensions_change_contact_time(parameter_adapter, authoring):
    """SHAPE-02/FIX-SHAPE-CONTACT: collider dimensions change first-contact time."""
    baseline = _measure_first_contact_step(parameter_adapter, authoring, shape="box")
    changed = _measure_first_contact_step(parameter_adapter, authoring, shape="box", size=(0.2, 0.2, 0.4))
    case = _case(parameter_adapter, "SHAPE-02", authoring, rtol=0.0, atol=1.0)
    assert changed < baseline, case.message(changed, f"earlier than baseline step {baseline}")


@pytest.mark.parametrize("parameter_adapter", ["kamino", "mjwarp"], indirect=True)
def test_shape_02_runtime_scale_rejected(parameter_adapter):
    """SHAPE-02: scale changes after simulation start raise the documented error."""
    with _contact_scene(
        parameter_adapter,
        "cfg",
        object_shape="box",
        object_position=(0.0, 0.0, 0.8),
    ) as (_, _, _, env):
        with pytest.raises(RuntimeError, match="Randomizing scale while simulation is running"):
            randomize_rigid_body_scale(
                env,  # pyright: ignore[reportArgumentType]
                None,
                (1.5, 1.5),
                SceneEntityCfg("object"),
            )


@pytest.mark.parametrize("authoring", ["usd", "cfg"])
@pytest.mark.parametrize("parameter_adapter", _CONTACT_BACKENDS, indirect=True)
def test_shape_03_radius_changes_contact_time(parameter_adapter, authoring):
    """SHAPE-03/FIX-SHAPE-CONTACT: collision radius changes first-contact time."""
    baseline = _measure_first_contact_step(parameter_adapter, authoring, shape="sphere")
    changed = _measure_first_contact_step(parameter_adapter, authoring, shape="sphere", radius=0.2)
    case = _case(parameter_adapter, "SHAPE-03", authoring, rtol=0.0, atol=1.0)
    assert changed < baseline, case.message(changed, f"earlier than baseline step {baseline}")


_MJWARP_MARGIN_REASON = "newton-physics/newton#2106: production-default MuJoCo contacts zero Newton shape margins"


@pytest.mark.parametrize("authoring", ["usd", "cfg", "runtime"])
@pytest.mark.parametrize(
    "parameter_adapter",
    [
        pytest.param("kamino", id="kamino"),
        pytest.param(
            "mjwarp",
            id="mjwarp",
            marks=pytest.mark.xfail(strict=True, reason=_MJWARP_MARGIN_REASON),
        ),
    ],
    indirect=True,
)
def test_contact_01_margin_controls_resting_separation(parameter_adapter, authoring):
    """CONTACT-01/FIX-CONTACT-OFFSET: public offsets map to physical margin separation."""
    margin = 0.02
    contact_offset = 0.03
    with _contact_scene(
        parameter_adapter,
        authoring,
        object_shape="sphere",
        object_position=(0.0, 0.0, 0.8),
        rest_offset=margin,
        contact_offset=contact_offset,
    ) as (sim, body, sensor, _):
        for _ in range(360):
            _step(sim, body, sensor)
        separation = float(body.data.root_com_pos_w.torch[0, 2]) - CONTACT_SPHERE_RADIUS - _GROUND_TOP
        expected = 2.0 * margin
        case = _case(
            parameter_adapter,
            "CONTACT-01",
            authoring,
            rtol=0.0,
            atol=contact_separation_atol(0.0),
        )
        assert_physical_close(separation, expected, case)
