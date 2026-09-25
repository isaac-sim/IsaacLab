# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX IMU sensor.

Mirrors the structure of source/isaaclab_physx/test/sensors/test_imu.py
but runs kitless under uv run python -m pytest — no AppLauncher needed.
SimulationContext is instantiated directly (it does not require Kit), and
UsdFileCfg(usd_path=ISAAC_NUCLEUS_DIR/...) downloads Nucleus assets via
omni.client (which works standalone in Kit's Python).

Tests that load the PhysX pendulum URDF (``test_single_dof_pendulum`` and
``test_indirect_attachment``) are skipped pending a USD-converted pendulum
asset. URDF→USD conversion requires the Kit URDF importer extension, which
is not loaded under the direct uv run python runner.

Process-global wheel state: like the rigid-object test, this file mixes
procedural USD assets (``test_constant_velocity``, ``test_constant_acceleration``,
``test_attachment_validity``) with a Nucleus asset (``test_offset_calculation``). ``omni.client``
must be loaded before the first OVPhysX scene is torn down; otherwise a later
first import can fail native symbol resolution after ``ovphysx.reset()``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Wheel gate: skip the whole file if the ovphysx wheel is missing or too old.
# ---------------------------------------------------------------------------
import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")
_TT_module = pytest.importorskip(
    "isaaclab_ov.tensor_types",
    reason="isaaclab_ov.tensor_types not importable",
)
if not hasattr(_TT_module, "RIGID_BODY_POSE"):
    pytest.skip(
        "ovphysx wheel does not yet expose RIGID_BODY_POSE / RIGID_BODY_VELOCITY",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Imports (after wheel gate)
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxArticulationCfg  # noqa: E402

# Preload Omni Client while Kit's native libraries are still in a clean loader
# state. Importing it for the first time after an OVPhysX reset can fail with an
# undefined symbol from omni.client's native extension.
import omni.client  # noqa: E402,F401

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors.imu import Imu, ImuCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa: E402

wp.init()

pytestmark = pytest.mark.device_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ENVS = 2
"""Number of environment instances spawned in each test scene."""

# offset of imu_link from base_link on anymal_c
POS_OFFSET = (0.2488, 0.00835, 0.04628)
ROT_OFFSET = (0, 0, 0.7071068, 0.7071068)


# ---------------------------------------------------------------------------
# Scene-builder helpers (real backend, Nucleus / procedural USD assets)
# ---------------------------------------------------------------------------


def _spawn_envs(num_envs: int) -> None:
    """Create per-env Xform containers at ``/World/env_<i>``.

    These match the prim-path layout the IMU's attachment-validity test
    expects, and provide a parent for per-env asset spawns.
    """
    # /World/env_<i> Xforms are siblings under /World — no envs container needed
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i * 5.0, 0.0, 0.0))


def _spawn_balls(num_envs: int, height: float = 0.5) -> RigidObject:
    """Spawn a sphere rigid body at ``/World/env_<i>/ball`` for each env.

    Returns the :class:`RigidObject` whose binding pattern matches all spawned
    instances. The :class:`RigidObject` does the per-env spawning itself when
    ``spawn`` is set; we only have to create the env Xform containers first
    (handled by :func:`_spawn_envs`). The prim path is a regex; the ovphysx
    binding pattern underneath it is an fnmatch glob.
    """
    spawn_cfg = sim_utils.SphereCfg(
        radius=0.25,
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=0.5),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg = RigidObjectCfg(
        prim_path="/World/env_[^/]+/ball",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    return RigidObject(cfg)


def _spawn_cubes(num_envs: int, height: float = 0.5) -> RigidObject:
    """Spawn a cube rigid body at ``/World/env_<i>/cube`` for each env."""
    spawn_cfg = sim_utils.CuboidCfg(
        size=(0.25, 0.25, 0.25),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=0.5),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg = RigidObjectCfg(
        prim_path="/World/env_[^/]+/cube",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.0, height)),
    )
    return RigidObject(cfg)


def _spawn_anymal(num_envs: int) -> Articulation:
    """Spawn the Anymal-C articulation at ``/World/env_<i>/robot`` for each env.

    Uses :data:`~isaaclab_assets.robots.anymal.ANYMAL_C_CFG` directly so the
    actuator and init-state configuration matches the PhysX reference test.
    The :class:`Articulation` performs the per-env spawn itself once the env
    Xform containers exist; :func:`_spawn_envs` must be called first.
    """
    cfg = ANYMAL_C_CFG.replace(prim_path="/World/env_[^/]+/robot")
    cfg.init_state.pos = (0.0, 2.0, 1.0)
    # bump solver iteration counts to match the PhysX test's scene cfg -- the counts live on the
    # PhysX articulation fragment
    physx_articulation = next(frag for frag in cfg.spawn.articulation_props if isinstance(frag, PhysxArticulationCfg))
    physx_articulation.solver_position_iteration_count = 32
    physx_articulation.solver_velocity_iteration_count = 32
    return Articulation(cfg)


def _make_imu(prim_path: str, offset: ImuCfg.OffsetCfg | None = None) -> Imu:
    """Create an :class:`Imu` with the given prim path and optional offset."""
    cfg = ImuCfg(prim_path=prim_path)
    if offset is not None:
        cfg.offset = offset
    return Imu(cfg)


@configclass
class _StaleResetSceneCfg(InteractiveSceneCfg):
    """Minimal scene for the post-reset staleness regression test."""

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.5),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
    )
    imu_cube: ImuCfg = ImuCfg(prim_path="{ENV_REGEX_NS}/cube")


# ---------------------------------------------------------------------------
# Process-global device-mode lock (matches the rigid-object and contact-sensor
# tests). The ovphysx wheel can only run one device per process; parametrized
# tests skip on the unlocked device so single-device runs finish cleanly.
# ---------------------------------------------------------------------------

_LOCKED_DEVICE: list[str | None] = [None]


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
# Sim context fixture (real OVPhysX backend, device-parametrized)
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_ctx(device: str):
    """Build an OVPhysX-backed :class:`SimulationContext` on the requested device.

    Yields:
        The simulation context, set up with a small fixed timestep matching
        the PhysX reference test (``dt=0.001``) for IMU numerical-differentiation
        accuracy.
    """
    with build_simulation_context(
        sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, dt=0.001),
    ) as sim:
        yield sim


_DEVICES = ["cpu", "cuda:0"]


# ===========================================================================
# Constant-velocity / constant-acceleration tests (rigid bodies)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
def test_constant_velocity(sim_ctx, device):
    """Test the IMU sensor with a constant velocity.

    Expected behavior is that the linear acceleration is approximately the
    same at every time step: in each step we set the same velocity, so the
    finite-difference derivative settles to zero (plus the gravity bias).
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    cubes = _spawn_cubes(NUM_ENVS)
    imu_ball = _make_imu("/World/env_[^/]+/ball")
    imu_cube = _make_imu("/World/env_[^/]+/cube")
    sim_ctx.reset()

    prev_lin_acc_ball = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=device)
    prev_lin_acc_cube = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=device)

    for idx in range(200):
        # set velocity
        velocity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        )
        balls.write_root_velocity_to_sim(velocity)
        cubes.write_root_velocity_to_sim(velocity)
        # write data to sim
        balls.write_data_to_sim()
        cubes.write_data_to_sim()
        # perform step
        sim_ctx.step()
        # read data from sim
        dt = sim_ctx.get_physics_dt()
        balls.update(dt)
        cubes.update(dt)
        imu_ball.update(dt, force_recompute=True)
        imu_cube.update(dt, force_recompute=True)

        if idx > 1:
            # check the imu accelerations
            torch.testing.assert_close(
                imu_ball.data.lin_acc_b.torch,
                prev_lin_acc_ball,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                imu_cube.data.lin_acc_b.torch,
                prev_lin_acc_cube,
                rtol=1e-3,
                atol=1e-3,
            )

        # update previous values
        prev_lin_acc_ball = imu_ball.data.lin_acc_b.torch.clone()
        prev_lin_acc_cube = imu_cube.data.lin_acc_b.torch.clone()


@pytest.mark.parametrize("device", _DEVICES)
def test_constant_acceleration(sim_ctx, device):
    """A constant applied force yields the solver acceleration F/m.

    The IMU reports proper acceleration, so for a ball that is otherwise in free fall the
    ``-g`` of the fall cancels the ``+g`` accelerometer bias and only ``F/m`` remains.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_[^/]+/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    # Pick the target acceleration and derive the force from the simulated mass, so the
    # expectation stays correct if the spawn config changes.
    expected_acc = 0.5  # [m/s^2]
    ball_mass = balls.data.body_mass.torch[:, 0]
    external_wrench_b = torch.zeros((NUM_ENVS, 1, 6), device=device)
    external_wrench_b[:, 0, 0] = ball_mass * expected_acc
    balls.permanent_wrench_composer.set_forces_and_torques_index(
        forces=external_wrench_b[..., :3],
        torques=external_wrench_b[..., 3:],
    )

    # keep the window short: the kitless scene has no ground, so the ball simply falls
    for idx in range(10):
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        imu_ball.update(dt, force_recompute=True)

        # skip first step where the solver has not integrated the force yet
        if idx < 1:
            continue

        # check the imu linear acceleration data (gravity cancels in free fall)
        torch.testing.assert_close(
            imu_ball.data.lin_acc_b.torch,
            math_utils.quat_apply_inverse(
                balls.data.root_quat_w.torch,
                torch.tensor([[expected_acc, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1),
            ),
            rtol=1e-4,
            atol=1e-4,
        )

        # check the angular velocity
        torch.testing.assert_close(
            imu_ball.data.ang_vel_b.torch,
            balls.data.root_ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )

    s = str(imu_ball)
    assert "Imu sensor @ '/World/env_[^/]+/ball'" in s
    assert "binding pattern" in s
    assert "/World/env_[^/]+/ball" in s
    assert "number of sensors : 2" in s


# ===========================================================================
# Articulation tests (anymal-C, USD asset from Nucleus)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
def test_offset_calculation(sim_ctx, device):
    """Test offset configuration argument.

    Two IMUs on the anymal-C robot — one at ``base`` with a configured offset
    matching the location of ``imu_link``, and one directly at ``imu_link``
    (a non-physics child of the ``base`` rigid body) — should resolve the same
    offset and produce identical readings.
    """
    _spawn_envs(NUM_ENVS)
    robot = _spawn_anymal(NUM_ENVS)
    imu_robot_imu_link = _make_imu("/World/env_[^/]+/robot/base/imu_link")
    imu_robot_base = _make_imu(
        "/World/env_[^/]+/robot/base",
        offset=ImuCfg.OffsetCfg(pos=POS_OFFSET, rot=ROT_OFFSET),
    )
    sim_ctx.reset()

    torch.testing.assert_close(
        wp.to_torch(imu_robot_imu_link._offset_pos_b),
        wp.to_torch(imu_robot_base._offset_pos_b),
    )
    torch.testing.assert_close(
        wp.to_torch(imu_robot_imu_link._offset_quat_b),
        wp.to_torch(imu_robot_base._offset_quat_b),
        rtol=1e-4,
        atol=1e-4,
    )

    dt = sim_ctx.get_physics_dt()

    for idx in range(500):
        # apply increasing root velocity
        velocity = torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        robot.write_root_velocity_to_sim(velocity)
        robot.write_data_to_sim()
        sim_ctx.step()
        robot.update(dt)
        imu_robot_imu_link.update(dt, force_recompute=True)
        imu_robot_base.update(dt, force_recompute=True)

        # skip first step where initial velocity is zero
        if idx < 1:
            continue

        torch.testing.assert_close(
            imu_robot_base.data.lin_acc_b.torch,
            imu_robot_imu_link.data.lin_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            imu_robot_base.data.ang_vel_b.torch,
            imu_robot_imu_link.data.ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_reset(sim_ctx, device):
    """Test that ``reset`` zeroes out the IMU output and previous-velocity buffers.

    Mirrors the Newton ``test_reset`` parity check: drive the IMU until its
    buffers hold non-zero data, reset one env through ``env_ids`` and then all
    of them, and assert the raw warp buffers are zero where reset. We read the
    raw warp arrays directly because accessing ``imu.data`` triggers a lazy
    re-fill that masks reset bugs.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_[^/]+/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    # Drive both outputs non-zero: a spin for the gyro, and an applied force for the
    # accelerometer (a freely falling ball would read zero proper acceleration).
    ball_mass = balls.data.body_mass.torch[:, 0]
    external_wrench_b = torch.zeros((NUM_ENVS, 1, 6), device=device)
    external_wrench_b[:, 0, 0] = ball_mass  # 1 m/s^2 along x
    balls.permanent_wrench_composer.set_forces_and_torques_index(
        forces=external_wrench_b[..., :3],
        torques=external_wrench_b[..., 3:],
    )
    nonzero_vel = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1)
    balls.write_root_velocity_to_sim_index(root_velocity=nonzero_vel)
    for _ in range(5):
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        imu_ball.update(dt, force_recompute=True)

    # Buffers should hold non-zero state before reset.
    assert torch.any(wp.to_torch(imu_ball._data._lin_acc_b) != 0), "expected non-zero data before reset"
    assert torch.any(wp.to_torch(imu_ball._data._ang_vel_b) != 0), "expected non-zero data before reset"
    assert torch.any(imu_ball.data.lin_acc_b.torch[1] != 0), "expected env 1 to have non-zero data before reset"

    # reset only env 1
    imu_ball.reset(env_ids=[1])
    torch.testing.assert_close(
        wp.to_torch(imu_ball._data._lin_acc_b)[1],
        torch.zeros(3, dtype=torch.float32, device=device),
    )
    torch.testing.assert_close(
        wp.to_torch(imu_ball._data._ang_vel_b)[1],
        torch.zeros(3, dtype=torch.float32, device=device),
    )
    assert torch.any(wp.to_torch(imu_ball._data._lin_acc_b)[0] != 0), "env 0 should not be reset"

    imu_ball.reset()

    # Read raw warp buffers directly — ``imu.data`` would trigger a lazy re-fill that
    # bypasses the reset.
    ang_vel_after = wp.to_torch(imu_ball._data._ang_vel_b)
    lin_acc_after = wp.to_torch(imu_ball._data._lin_acc_b)
    torch.testing.assert_close(ang_vel_after, torch.zeros_like(ang_vel_after))
    torch.testing.assert_close(lin_acc_after, torch.zeros_like(lin_acc_after))


@pytest.mark.parametrize("device", _DEVICES)
def test_no_stale_data_after_scene_reset(sim_ctx, device):
    """Test ``scene.reset(env_ids)`` does not expose stale native velocity through ``imu.data``."""
    scene_cfg = _StaleResetSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=False)
    scene = InteractiveScene(scene_cfg)
    sim_ctx.reset()
    scene.reset()

    sensor: Imu = scene["imu_cube"]

    # Drive the native rigid-body velocity buffer non-zero. A freely falling body reads zero
    # proper acceleration, so spin the cube and assert on the gyro output, which a stale
    # refetch after the reset would surface again.
    cube: RigidObject = scene["cube"]
    nonzero_vel = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    cube.write_root_velocity_to_sim_index(root_velocity=nonzero_vel)
    scene.write_data_to_sim()
    sim_ctx.step()
    scene.update(dt=sim_ctx.get_physics_dt())

    assert torch.any(sensor.data.ang_vel_b.torch != 0), "expected non-zero sensor output before reset"

    # Reset without another physics step. The public accessor must keep reset outputs
    # instead of lazy-refetching stale native velocity.
    scene.reset(env_ids=torch.tensor([0], device=device))

    post_reset_lin_acc = sensor.data.lin_acc_b.torch
    post_reset_ang_vel = sensor.data.ang_vel_b.torch
    torch.testing.assert_close(post_reset_lin_acc, torch.zeros_like(post_reset_lin_acc))
    torch.testing.assert_close(post_reset_ang_vel, torch.zeros_like(post_reset_ang_vel))


# ===========================================================================
# Validation tests (no asset state required)
# ===========================================================================


# The shared SensorBase resolver raises before any device work, so the CPU pass covers it.
@pytest.mark.parametrize("device", ["cpu"])
def test_attachment_validity(sim_ctx, device):
    """Test invalid IMU attachment.

    An IMU cannot be attached directly to the world Xform — it must have a
    rigid-body ancestor in its prim tree.
    """
    _spawn_envs(NUM_ENVS)
    sim_ctx.reset()

    imu_world_cfg = ImuCfg(prim_path="/World/env_0")
    with pytest.raises(RuntimeError) as exc_info:
        imu_world = Imu(imu_world_cfg)
        imu_world._initialize_impl()
    assert exc_info.type is RuntimeError and "find a rigid body ancestor prim" in str(exc_info.value)


# ===========================================================================
# URDF-dependent tests — skipped pending USD pendulum asset
# ===========================================================================


@pytest.mark.skip(
    reason=(
        "Blocked on a USD-converted pendulum asset (the PhysX test loads"
        " source/isaaclab_physx/test/sensors/urdfs/simple_2_link.urdf via the Kit URDF importer,"
        " which is not loaded under the direct uv run python runner). Re-enable"
        " once a pre-converted USD pendulum is available."
    )
)
def test_single_dof_pendulum():
    """Test imu against analytical pendulum problem."""
    # If this test is ever un-skipped without porting the PhysX assertions, fail
    # explicitly rather than passing vacuously.
    pytest.fail(
        "test_single_dof_pendulum was un-skipped without a body — port the assertions from"
        " source/isaaclab_physx/test/sensors/test_imu.py::test_single_dof_pendulum."
    )


@pytest.mark.skip(
    reason=(
        "Blocked on a USD-converted pendulum asset (the PhysX test loads"
        " source/isaaclab_physx/test/sensors/urdfs/simple_2_link.urdf via the Kit URDF importer,"
        " which is not loaded under the direct uv run python runner). Re-enable"
        " once a pre-converted USD pendulum is available."
    )
)
def test_indirect_attachment():
    """Test attaching the IMU through an Xform primitive offset chain."""
    # If this test is ever un-skipped without porting the PhysX assertions, fail
    # explicitly rather than passing vacuously.
    pytest.fail(
        "test_indirect_attachment was un-skipped without a body — port the assertions from"
        " source/isaaclab_physx/test/sensors/test_imu.py::test_indirect_attachment."
    )
