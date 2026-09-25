# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "sensors"))

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors.joint_wrench import joint_wrench_sensor as joint_wrench_module
from isaaclab_physx.sensors.joint_wrench.joint_wrench_sensor import JointWrenchSensor as PhysxJointWrenchSensor
from isaaclab_physx.sensors.joint_wrench.joint_wrench_sensor_data import JointWrenchSensorData
from isaaclab_physx.sim.schemas import PhysxJointCfg
from joint_wrench_contract import test_joint_wrench_frame  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensor, JointWrenchSensorCfg
from isaaclab.sensors.joint_wrench import BaseJointWrenchSensor
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.ant import ANT_CFG


def _make_single_joint_articulation_cfg() -> ArticulationCfg:
    """Single-joint revolute test articulation (root ``CenterPivot`` + arm ``Arm``)."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
            joint_drive_props=[sim_utils.UsdPhysicsDriveCfg(max_force=80.0), PhysxJointCfg(max_joint_velocity=5.0)],
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


@configclass
class _SingleJointSceneCfg(InteractiveSceneCfg):
    """Scene with a single-joint articulation and the joint-wrench sensor."""

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = _make_single_joint_articulation_cfg()
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _NestedRootAntSceneCfg(InteractiveSceneCfg):
    """Ant USD asset whose articulation root is nested under the configured asset prim."""

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@pytest.fixture
def sim():
    """Simulation context using the PhysX backend."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        physics=PhysxCfg(),
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None
        yield sim_ctx


def _physx_incoming_joint_wrench(sensor: JointWrenchSensor) -> torch.Tensor:
    """Read the raw PhysX incoming joint wrench tensor.

    PhysX reports spatial vectors as force followed by torque. Shape is
    ``(num_envs, num_bodies, 6)``.
    """
    raw_wrench = sensor._root_view.get_link_incoming_joint_force().view(wp.spatial_vectorf)
    return wp.to_torch(raw_wrench)


def _assert_sensor_matches_physx_tensor(sensor: JointWrenchSensor) -> None:
    """The sensor exposes the PhysX tensor's existing child-joint-frame components."""
    raw_wrench = _physx_incoming_joint_wrench(sensor)
    torch.testing.assert_close(sensor.data.force.torch, raw_wrench[..., :3])
    torch.testing.assert_close(sensor.data.torque.torch, raw_wrench[..., 3:])


# ---------------------------------------------------------------------------
# Sensor data — pre-init contract
# ---------------------------------------------------------------------------


def test_data_before_init_is_none():
    """``force``/``torque`` return ``None`` before :meth:`create_buffers` runs."""
    from isaaclab_physx.sensors.joint_wrench import JointWrenchSensorData

    data = JointWrenchSensorData()
    assert data.force is None
    assert data.torque is None


# ---------------------------------------------------------------------------
# Initialization and shapes
# ---------------------------------------------------------------------------


def test_initialization_and_shapes(sim):
    """Sensor initializes on sim reset and exposes correctly-shaped buffers."""
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=2))
    sim.reset()

    robot: Articulation = scene["robot"]
    sensor: JointWrenchSensor = scene["wrench"]
    sim.step()
    scene.update(sim.get_physics_dt())

    # PhysX reports one incoming joint wrench per articulation link, including the root link.
    num_envs = 2
    num_bodies = robot.num_bodies
    assert sensor.data.force.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.data.torque.torch.shape == (num_envs, num_bodies, 3)
    assert sensor.body_names == robot.body_names
    assert sensor.find_bodies("Arm") == ([robot.body_names.index("Arm")], ["Arm"])
    assert sensor._root_view is robot.root_view  # noqa: SLF001
    _assert_sensor_matches_physx_tensor(sensor)
    sensor_str = str(sensor)
    assert "physx" in sensor_str
    assert "Joint wrench sensor" in sensor_str


def test_nested_articulation_root_resolution(sim):
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
    _assert_sensor_matches_physx_tensor(sensor)


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


def test_reset_with_env_ids_only_zeros_selected_envs(sim):
    """Partial reset via env_ids should zero the selected envs and preserve the others; a full reset zeros all."""
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

    sensor.reset()

    # Access raw buffers to skip lazy re-population from the PhysX view on the next data read.
    force_after = wp.to_torch(sensor._data._force)
    torque_after = wp.to_torch(sensor._data._torque)
    torch.testing.assert_close(force_after, torch.zeros_like(force_after))
    torch.testing.assert_close(torque_after, torch.zeros_like(torque_after))


def test_no_stale_data_after_scene_reset(sim):
    """Regression for #4970: ``scene.reset(env_ids)`` must not surface pre-reset wrenches.

    Reproduces the manager-based RL flow where an environment terminates,
    ``_reset_idx`` runs, and the next observation read happens before any further physics
    step. The joint-wrench sensor's lazy refetch via :attr:`data` must not return PhysX's
    stale post-step buffer for the reset env.
    """
    scene = InteractiveScene(_SingleJointSceneCfg(num_envs=1))
    sim.reset()

    sensor: JointWrenchSensor = scene["wrench"]
    for _ in range(100):
        sim.step()
        scene.update(sim.get_physics_dt())

    # Sanity: wrench is non-zero (joint resists gravity on the arm).
    pre_reset_force = sensor.data.force.torch.clone()
    pre_reset_torque = sensor.data.torque.torch.clone()
    assert torch.any(pre_reset_force != 0) or torch.any(pre_reset_torque != 0), "Expected non-zero wrench before reset"

    # Reset via the scene (mirrors what ``ManagerBasedRLEnv._reset_idx`` does).
    scene.reset(env_ids=torch.tensor([0], device=sensor.device))

    # The lazy public ``data`` accessor must not refetch a stale PhysX buffer here.
    post_reset_force = sensor.data.force.torch
    post_reset_torque = sensor.data.torque.torch
    torch.testing.assert_close(post_reset_force, torch.zeros_like(post_reset_force))
    torch.testing.assert_close(post_reset_torque, torch.zeros_like(post_reset_torque))


class _FakeArticulationView:
    """Return one stable PhysX-like wrench buffer while counting typed-view construction."""

    def __init__(self, wrenches: wp.array):
        self.wrenches = wrenches
        self.get_count = 0
        self.view_count = 0

    def get_link_incoming_joint_force(self):
        self.get_count += 1
        return self

    def view(self, dtype):
        assert dtype == wp.spatial_vectorf
        self.view_count += 1
        return self.wrenches

    @property
    def ptr(self):
        return self.wrenches.ptr


def _make_joint_wrench_sensor(use_recorded_launch: bool = True, num_envs: int = 1):
    """Create a JointWrench sensor without a USD scene."""
    device = "cuda:0"
    wrenches_torch = torch.arange(1, num_envs * 6 + 1, dtype=torch.float32, device=device).reshape(num_envs, 1, 6)
    wrenches = wp.from_torch(wrenches_torch.contiguous()).view(wp.spatial_vectorf)
    root_view = _FakeArticulationView(wrenches)

    sensor = PhysxJointWrenchSensor.__new__(PhysxJointWrenchSensor)
    sensor.cfg = SimpleNamespace(prim_path="/World/Robot")
    sensor._device = device
    sensor._num_envs = num_envs
    sensor._num_bodies = 1
    sensor._root_view = root_view
    sensor._timestamp = wp.ones(num_envs, dtype=wp.float32, device=device)
    sensor._data = JointWrenchSensorData()
    sensor._data.create_buffers(num_envs=num_envs, num_bodies=1, device=device)
    sensor._raw_incoming_joint_wrench = None
    sensor._update_cmd = None
    sensor._use_recorded_launch = use_recorded_launch
    sensor._physics_sim_view = None
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None

    env_mask = wp.ones(num_envs, dtype=wp.bool, device=device)
    return sensor, root_view, wrenches_torch, env_mask


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_joint_wrench_records_and_replays_update_launch():
    """A recorded update should replay using changes to the stable environment mask."""
    sensor, _, wrenches_torch, env_mask = _make_joint_wrench_sensor(num_envs=2)
    env_mask_torch = wp.to_torch(env_mask)
    env_mask_torch[:] = torch.tensor([True, False], device=sensor.device)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)
    update_cmd = sensor._update_cmd

    assert update_cmd is not None
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[0, 0],
        torch.tensor([1.0, 2.0, 3.0], device=sensor.device),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._data._torque)[0, 0],
        torch.tensor([4.0, 5.0, 6.0], device=sensor.device),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[1, 0],
        torch.zeros(3, device=sensor.device),
    )

    wrenches_torch[0, 0, 0] = 101.0
    wrenches_torch[1, 0, 0] = 107.0
    env_mask_torch[:] = torch.tensor([False, True], device=sensor.device)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert sensor._update_cmd is update_cmd
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[0, 0],
        torch.tensor([1.0, 2.0, 3.0], device=sensor.device),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[1, 0],
        torch.tensor([107.0, 8.0, 9.0], device=sensor.device),
    )


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_joint_wrench_falls_back_when_recording_fails(monkeypatch):
    """A recording failure should disable recording and execute the current update eagerly.

    Later eager updates refresh the PhysX buffer but reuse one typed view over it.
    """
    sensor, root_view, _, env_mask = _make_joint_wrench_sensor()
    original_launch = joint_wrench_module.wp.launch

    def launch_with_recording_failure(*args, record_cmd=False, **kwargs):
        if record_cmd:
            raise RuntimeError("recording failed")
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(joint_wrench_module.wp, "launch", launch_with_recording_failure)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert not sensor._use_recorded_launch
    torch.testing.assert_close(
        wp.to_torch(sensor._data._force)[0, 0],
        torch.tensor([1.0, 2.0, 3.0], device=sensor.device),
    )

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert root_view.get_count == 2
    assert root_view.view_count == 1


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_joint_wrench_invalidation_drops_cached_launch_state(monkeypatch):
    """Physics invalidation should release the cached PhysX view and recorded command."""
    sensor, _, _, _ = _make_joint_wrench_sensor()
    sensor._raw_incoming_joint_wrench = object()
    sensor._update_cmd = object()
    monkeypatch.setattr(BaseJointWrenchSensor, "_invalidate_initialize_callback", lambda self, event: None)

    sensor._invalidate_initialize_callback(None)

    assert sensor._root_view is None
    assert sensor._raw_incoming_joint_wrench is None
    assert sensor._update_cmd is None
