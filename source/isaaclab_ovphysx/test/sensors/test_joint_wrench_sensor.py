# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX JointWrenchSensor.

Mirrors ``isaaclab_physx`` ``test_joint_wrench_sensor.py``; only the
fixtures and raw-tensor read helper change. Physical assertions are
kept byte-identical so the two backends report the same convention.

``ovphysx<=0.3.7`` binds device mode (CPU vs GPU) at the C++ layer on the
first ``ovphysx.PhysX(device=...)`` construction.  Full coverage therefore
requires two pytest runs -- once with ``-k 'cpu'`` and once with
``-k 'cuda:0'``.  The ``_ovphysx_skip_other_device`` autouse fixture below
preempts the manager's :exc:`RuntimeError` by ``pytest.skip``-ing on the
unlocked device so single-device runs finish cleanly.
"""

from __future__ import annotations

import math

import pytest
import torch
import warp as wp

from pxr import Gf, UsdPhysics

# The CI isaaclab_ov* pattern unintentionally collects isaaclab_ovphysx tests,
# but the ovphysx wheel is not installed in that environment. Skip gracefully
# so the isaaclab_ov CI pipeline is not blocked by an unrelated dependency.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import JointWrenchSensor, JointWrenchSensorCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from isaaclab_assets.robots.ant import ANT_CFG  # noqa: E402

wp.init()

# OVPhysX/Warp and the PyTorch reference use different float32 operation order on CUDA.
_OVPHYSX_WRENCH_RTOL = 5e-6
_OVPHYSX_WRENCH_ATOL = 1e-5

# ---------------------------------------------------------------------------
# Device-lock autouse fixture (copied from test_contact_sensor.py)
# ---------------------------------------------------------------------------

_LOCKED_DEVICE: list[str | None] = [None]
"""Device the session pins to on the first parametrized test that runs."""


@pytest.fixture(autouse=True)
def _ovphysx_skip_other_device(request):
    """Skip parametrized tests on the device the session is not pinned to."""
    callspec = getattr(request.node, "callspec", None)
    device = callspec.params.get("device") if callspec is not None else None
    if device is None:
        return
    locked = _LOCKED_DEVICE[0]
    if locked is None:
        _LOCKED_DEVICE[0] = device
        return
    if device != locked:
        pytest.skip(
            f"ovphysx process-global device lock is held by '{locked}'; cannot run '{device}' "
            "tests in the same session.  Run pytest twice (once per device) for full coverage."
        )


# ---------------------------------------------------------------------------
# Simulation context helper (mirrors test_contact_sensor.py)
# ---------------------------------------------------------------------------


def _ovphysx_sim_context(device: str, **kwargs):
    """Wrapper around :func:`build_simulation_context` that injects OVPhysX cfg."""
    dt = kwargs.pop("dt", 1.0 / 120.0)
    gravity_enabled = kwargs.pop("gravity_enabled", True)
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=dt, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


# ---------------------------------------------------------------------------
# Scene configurations (copied from the PhysX test verbatim)
# ---------------------------------------------------------------------------


def _make_single_joint_articulation_cfg() -> ArticulationCfg:
    """Single-joint revolute test articulation (root ``CenterPivot`` + arm ``Arm``)."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=2000.0,
                damping=100.0,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _make_cartpole_articulation_cfg(pole_damping: float = 0.0) -> ArticulationCfg:
    """Two-joint cartpole articulation (cart + pole)."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
        ),
        actuators={
            "cart_actuator": ImplicitActuatorCfg(
                joint_names_expr=["slider_to_cart"], effort_limit_sim=400.0, stiffness=0.0, damping=10.0
            ),
            "pole_actuator": ImplicitActuatorCfg(
                joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=pole_damping
            ),
        },
    )


@configclass
class _SingleJointSceneCfg(InteractiveSceneCfg):
    """Scene with a single-joint articulation and the joint-wrench sensor."""

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = _make_single_joint_articulation_cfg()
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _CartpoleSceneCfg(InteractiveSceneCfg):
    """Scene with a cartpole (2-joint) articulation and the joint-wrench sensor."""

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = _make_cartpole_articulation_cfg()
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _CartpoleDampedSceneCfg(InteractiveSceneCfg):
    """Cartpole with pole damping for steady-state physics validation tests."""

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = _make_cartpole_articulation_cfg(pole_damping=10.0)
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _NestedRootAntSceneCfg(InteractiveSceneCfg):
    """Ant USD asset whose articulation root is nested under the configured asset prim."""

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@pytest.fixture
def sim(device):
    """Simulation context using the OVPhysX backend."""
    with _ovphysx_sim_context(device) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None
        yield sim_ctx


# ---------------------------------------------------------------------------
# Raw-tensor helpers
# ---------------------------------------------------------------------------


def _ovphysx_incoming_joint_wrench(sensor: JointWrenchSensor) -> torch.Tensor:
    """Read the raw OVPhysX incoming joint wrench tensor.

    OVPhysX reports spatial vectors as force followed by torque. Shape is
    ``(num_envs, num_bodies, 6)``. The read targets an independent scratch
    buffer so a subsequent ``sensor.data`` access (which re-reads into the
    sensor's own ``_wrench_buf``) cannot retroactively change the returned
    snapshot — mirrors how the PhysX helper returns an independent array from
    :meth:`ArticulationView.get_link_incoming_joint_force`.
    """
    scratch_buf = wp.zeros((sensor._num_envs, sensor._num_bodies), dtype=wp.spatial_vectorf, device=sensor._device)
    scratch_view = wp.array(
        ptr=scratch_buf.ptr,
        shape=sensor._wrench_binding.shape,
        dtype=wp.float32,
        device=str(scratch_buf.device),
        copy=False,
    )
    sensor._wrench_binding.read(scratch_view)
    return wp.to_torch(scratch_buf)


def _assert_sensor_matches_ovphysx_tensor(sensor: JointWrenchSensor) -> None:
    """Compare sensor buffers to the raw OVPhysX tensor transformed into joint frames."""
    raw_wrench = _ovphysx_incoming_joint_wrench(sensor)
    sensor_data = sensor.data

    expected_force, expected_torque = _ovphysx_incoming_joint_wrench_in_joint_frame(sensor, raw_wrench)
    torch.testing.assert_close(
        sensor_data.force.torch, expected_force, rtol=_OVPHYSX_WRENCH_RTOL, atol=_OVPHYSX_WRENCH_ATOL
    )
    torch.testing.assert_close(
        sensor_data.torque.torch, expected_torque, rtol=_OVPHYSX_WRENCH_RTOL, atol=_OVPHYSX_WRENCH_ATOL
    )


def _ovphysx_incoming_joint_wrench_in_joint_frame(
    sensor: JointWrenchSensor, raw_wrench: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform raw OVPhysX body-frame incoming joint wrenches into the configured convention."""
    force_b = raw_wrench[..., :3]
    torque_b = raw_wrench[..., 3:]
    joint_pos_b = wp.to_torch(sensor._joint_pos_b).unsqueeze(0)
    joint_quat_b = wp.to_torch(sensor._joint_quat_b).unsqueeze(0)
    torque_joint_anchor_b = torque_b - torch.cross(joint_pos_b.expand_as(force_b), force_b, dim=-1)

    flat_joint_quat_b = joint_quat_b.expand_as(raw_wrench[..., :4]).reshape(-1, 4)
    expected_force = math_utils.quat_apply_inverse(flat_joint_quat_b, force_b.reshape(-1, 3)).reshape(force_b.shape)
    expected_torque = math_utils.quat_apply_inverse(flat_joint_quat_b, torque_joint_anchor_b.reshape(-1, 3)).reshape(
        torque_b.shape
    )
    return expected_force, expected_torque


def _set_child_joint_frame(scene: InteractiveScene, child_body_name: str) -> None:
    """Set a non-identity child-side joint frame for the requested body in env 0."""
    for prim in scene.stage.Traverse():
        if not prim.GetPath().pathString.startswith("/World/envs/env_0/Robot"):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint and any(target.name == child_body_name for target in joint.GetBody1Rel().GetTargets()):
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.25, -0.15, 0.1))
            joint.GetLocalRot1Attr().Set(
                Gf.Quatf(
                    math.cos(math.pi / 4.0),
                    Gf.Vec3f(math.sin(math.pi / 4.0), 0.0, 0.0),
                )
            )
            return
    raise RuntimeError(f"Failed to find a USD joint with child body '{child_body_name}'.")


# ---------------------------------------------------------------------------
# Sensor data — pre-init contract
# ---------------------------------------------------------------------------


def test_data_before_init_is_none():
    """``force``/``torque`` return ``None`` before :meth:`create_buffers` runs."""
    from isaaclab_ovphysx.sensors.joint_wrench import JointWrenchSensorData

    data = JointWrenchSensorData()
    assert data.force is None
    assert data.torque is None


# ---------------------------------------------------------------------------
# Initialization and shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_initialization_and_shapes(sim, device):
    """Sensor initializes on sim reset and exposes correctly-shaped buffers."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    num_envs = 2
    num_bodies = robot.num_bodies
    assert sensor.data.force.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.body_names == robot.body_names
    assert sensor.find_bodies("Arm") == ([robot.body_names.index("Arm")], ["Arm"])
    _assert_sensor_matches_ovphysx_tensor(sensor)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_multi_body_articulation(sim, device):
    """Cartpole exposes a wrench for each link labelled by body name."""
    scene = InteractiveScene(_CartpoleSceneCfg(num_envs=2))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    num_envs = 2
    num_bodies = robot.num_bodies
    assert sensor.data.force.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.body_names == robot.body_names
    assert len(sensor.body_names) == num_bodies
    _assert_sensor_matches_ovphysx_tensor(sensor)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_nested_articulation_root_resolution(sim, device):
    """Sensor accepts an asset prim path whose articulation root is nested in the USD asset."""
    scene = InteractiveScene(_NestedRootAntSceneCfg(num_envs=1))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    assert sensor.body_names == robot.body_names
    assert sensor.data.force.torch.shape == (1, robot.num_bodies, 3)
    assert sensor.data.torque.torch.shape == (1, robot.num_bodies, 3)
    _assert_sensor_matches_ovphysx_tensor(sensor)


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_force_and_torque_components_at_rest(sim, device):
    """Component-level validation of force and torque against the OVPhysX tensor API."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    for _ in range(400):
        sim.step()
        scene.update(sim.get_physics_dt())

    _assert_sensor_matches_ovphysx_tensor(sensor)

    arm_idx = robot.body_names.index("Arm")
    raw_wrench = _ovphysx_incoming_joint_wrench(sensor)
    assert torch.any(raw_wrench[:, arm_idx, :] != 0.0)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_non_identity_joint_frame_transform(sim, device):
    """OVPhysX raw body-frame wrench is converted to the child-side joint frame."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    _set_child_joint_frame(scene, "Arm")
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    arm_idx = robot.body_names.index("Arm")

    for _ in range(400):
        sim.step()
        scene.update(sim.get_physics_dt())

    raw_wrench = _ovphysx_incoming_joint_wrench(sensor)
    expected_force, expected_torque = _ovphysx_incoming_joint_wrench_in_joint_frame(sensor, raw_wrench)
    torch.testing.assert_close(
        sensor.data.force.torch, expected_force, rtol=_OVPHYSX_WRENCH_RTOL, atol=_OVPHYSX_WRENCH_ATOL
    )
    torch.testing.assert_close(
        sensor.data.torque.torch, expected_torque, rtol=_OVPHYSX_WRENCH_RTOL, atol=_OVPHYSX_WRENCH_ATOL
    )

    raw_force = raw_wrench[:, arm_idx, :3]
    raw_torque = raw_wrench[:, arm_idx, 3:]
    assert not torch.allclose(sensor.data.force.torch[:, arm_idx], raw_force)
    assert not torch.allclose(sensor.data.torque.torch[:, arm_idx], raw_torque)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_wrench_with_external_force_and_torque(sim, device):
    """Full wrench validation with external force and torque applied."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]
    arm_idx = robot.body_names.index("Arm")

    ext_force_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_force_b[:, arm_idx, 1] = 10.0
    ext_torque_b = torch.zeros((1, robot.num_bodies, 3), device=sim.device)
    ext_torque_b[:, arm_idx, 2] = 10.0

    for _ in range(800):
        robot.permanent_wrench_composer.set_forces_and_torques_index(forces=ext_force_b, torques=ext_torque_b)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    _assert_sensor_matches_ovphysx_tensor(sensor)

    raw_wrench = _ovphysx_incoming_joint_wrench(sensor)
    assert torch.any(raw_wrench[:, arm_idx, :] != 0.0)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_interior_joint_wrench_at_rest(sim, device):
    """Interior joint wrench matches the raw OVPhysX incoming-joint tensor."""
    scene = InteractiveScene(_CartpoleDampedSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    robot: Articulation = scene["robot"]

    for _ in range(800):
        sim.step()
        scene.update(sim.get_physics_dt())

    _assert_sensor_matches_ovphysx_tensor(sensor)

    cart_idx = robot.body_names.index("cart")
    raw_wrench = _ovphysx_incoming_joint_wrench(sensor)
    assert torch.any(raw_wrench[:, cart_idx, :] != 0.0)


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sensor_print(sim, device):
    """Test that the sensor string representation works."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    sensor_str = str(sensor)
    assert "ovphysx" in sensor_str
    assert "Joint wrench sensor" in sensor_str


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reset_zeros_buffers(sim, device):
    """Resetting the sensor clears the force / torque buffers."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    assert torch.any(sensor.data.force.torch != 0), "Expected non-zero data before reset"

    sensor.reset()

    force_after = wp.to_torch(sensor._data._force)
    torque_after = wp.to_torch(sensor._data._torque)
    torch.testing.assert_close(force_after, torch.zeros_like(force_after))
    torch.testing.assert_close(torque_after, torch.zeros_like(torque_after))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reset_with_env_ids_only_zeros_selected_envs(sim, device):
    """Partial reset via env_ids should zero the selected envs and preserve the others."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=4))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    force_before = sensor.data.force.torch.clone()
    assert torch.any(force_before != 0), "Expected non-zero data before reset"

    sensor.reset(env_ids=[0, 2])

    force_after = wp.to_torch(sensor._data._force)
    torch.testing.assert_close(force_after[0], torch.zeros_like(force_after[0]))
    torch.testing.assert_close(force_after[2], torch.zeros_like(force_after[2]))
    torch.testing.assert_close(force_after[1], force_before[1])
    torch.testing.assert_close(force_after[3], force_before[3])


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_no_stale_data_after_scene_reset(sim, device):
    """Regression for #4970: ``scene.reset(env_ids)`` must not surface pre-reset wrenches (OVPhysX)."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    pre_reset_force = sensor.data.force.torch.clone()
    pre_reset_torque = sensor.data.torque.torch.clone()
    assert torch.any(pre_reset_force != 0) or torch.any(pre_reset_torque != 0), "Expected non-zero wrench before reset"

    scene.reset(env_ids=torch.tensor([0], device=sensor.device))

    post_reset_force = sensor.data.force.torch
    post_reset_torque = sensor.data.torque.torch
    torch.testing.assert_close(post_reset_force, torch.zeros_like(post_reset_force))
    torch.testing.assert_close(post_reset_torque, torch.zeros_like(post_reset_torque))
