# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX IMU sensor.

Mirrors the structure of source/isaaclab_physx/test/sensors/test_imu.py
but runs kitless under ./isaaclab.sh -p -m pytest — no AppLauncher needed.
SimulationContext is instantiated directly (it does not require Kit), and
UsdFileCfg(usd_path=ISAAC_NUCLEUS_DIR/...) downloads Nucleus assets via
omni.client (which works standalone in Kit's Python).

Tests that load the PhysX pendulum URDF (``test_single_dof_pendulum`` and
``test_indirect_attachment``) are skipped pending a USD-converted pendulum
asset. URDF→USD conversion requires the Kit URDF importer extension, which
is not loaded under the direct ./isaaclab.sh -p runner.

Process-global wheel state: like the rigid-object test, this file mixes
procedural USD assets (``test_constant_velocity``, ``test_constant_acceleration``,
``test_attachment_validity``, ``test_sensor_print``) with Nucleus assets
(``test_offset_calculation``, ``test_env_ids_propagation``). ``omni.client``
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
    "isaaclab_ovphysx.tensor_types",
    reason="isaaclab_ovphysx.tensor_types not importable",
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
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

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
from isaaclab.utils.configclass import configclass  # noqa: E402

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
    (handled by :func:`_spawn_envs`). Note the ovphysx pattern uses an
    fnmatch glob (``env_*``), not a regex.
    """
    spawn_cfg = sim_utils.SphereCfg(
        radius=0.25,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg = RigidObjectCfg(
        prim_path="/World/env_*/ball",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    return RigidObject(cfg)


def _spawn_cubes(num_envs: int, height: float = 0.5) -> RigidObject:
    """Spawn a cube rigid body at ``/World/env_<i>/cube`` for each env."""
    spawn_cfg = sim_utils.CuboidCfg(
        size=(0.25, 0.25, 0.25),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cfg = RigidObjectCfg(
        prim_path="/World/env_*/cube",
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
    cfg = ANYMAL_C_CFG.replace(prim_path="/World/env_.*/robot")
    cfg.init_state.pos = (0.0, 2.0, 1.0)
    # bump solver iteration counts to match the PhysX test's scene cfg
    cfg.spawn.articulation_props.solver_position_iteration_count = 32
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 32
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
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
    imu_ball = _make_imu("/World/env_*/ball")
    imu_cube = _make_imu("/World/env_*/cube")
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
    """Test the IMU sensor with a constant acceleration."""
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()

    for idx in range(100):
        # set acceleration via increasing velocity per step
        velocity = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        balls.write_root_velocity_to_sim(velocity)
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        imu_ball.update(dt, force_recompute=True)

        # skip first step where initial velocity is zero
        if idx < 1:
            continue

        # check the imu linear acceleration data (includes gravity)
        torch.testing.assert_close(
            imu_ball.data.lin_acc_b.torch,
            math_utils.quat_apply_inverse(
                balls.data.root_quat_w.torch,
                torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1) / dt
                + torch.tensor([[0.0, 0.0, 9.81]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1),
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


# ===========================================================================
# Articulation tests (anymal-C, USD asset from Nucleus)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
def test_offset_calculation(sim_ctx, device):
    """Test offset configuration argument.

    Two IMUs on the anymal-C robot — one at ``base`` with a configured offset
    matching the location of ``imu_link``, and one directly at ``imu_link``
    — should produce identical readings.
    """
    _spawn_envs(NUM_ENVS)
    robot = _spawn_anymal(NUM_ENVS)
    imu_robot_imu_link = _make_imu("/World/env_*/robot/base/imu_link")
    imu_robot_base = _make_imu(
        "/World/env_*/robot/base",
        offset=ImuCfg.OffsetCfg(pos=POS_OFFSET, rot=ROT_OFFSET),
    )
    sim_ctx.reset()

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
def test_env_ids_propagation(sim_ctx, device):
    """Test that ``env_ids`` argument propagates through update and reset methods."""
    _spawn_envs(NUM_ENVS)
    robot = _spawn_anymal(NUM_ENVS)
    imu_robot_imu_link = _make_imu("/World/env_*/robot/base/imu_link")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()

    for idx in range(10):
        velocity = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        robot.write_root_velocity_to_sim(velocity)
        robot.write_data_to_sim()
        sim_ctx.step()
        robot.update(dt)
        imu_robot_imu_link.update(dt, force_recompute=True)

    # reset only env 1
    imu_robot_imu_link.reset(env_ids=[1])
    imu_robot_imu_link.update(dt, force_recompute=True)
    sim_ctx.step()
    imu_robot_imu_link.update(dt, force_recompute=True)


# ===========================================================================
# Physics-correctness sanity tests (ported from Newton)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
def test_sensor_initialization(sim_ctx, device):
    """Test that the OVPhysX IMU sensor initializes correctly."""
    _spawn_envs(NUM_ENVS)
    _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx.reset()

    assert imu_ball.num_instances == NUM_ENVS
    # Inspect the raw warp buffers directly — accessing ``imu.data`` triggers a
    # lazy FD-acceleration recompute that needs ``_dt`` (set by ``update``).
    assert imu_ball._data._ang_vel_b.shape == (NUM_ENVS,)
    assert imu_ball._data._lin_acc_b.shape == (NUM_ENVS,)
    assert imu_ball._data._ang_vel_b.dtype == wp.vec3f
    assert imu_ball._data._lin_acc_b.dtype == wp.vec3f


@pytest.mark.parametrize("device", _DEVICES)
def test_gravity_at_rest(sim_ctx, device):
    """Test that an IMU at rest measures gravity (~9.81 m/s^2 upward).

    Without InteractiveScene's terrain plumbing the ball falls forever, so we
    drive it kinematically: hold zero velocity for enough steps that the
    finite-difference acceleration of the *applied* velocity converges to zero
    and only the +g gravity bias remains.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    zero_vel = torch.zeros((NUM_ENVS, 6), dtype=torch.float32, device=device)
    for _ in range(5):
        balls.write_root_velocity_to_sim(zero_vel)
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        imu_ball.update(dt, force_recompute=True)

    lin_acc = imu_ball.data.lin_acc_b.torch
    torch.testing.assert_close(
        lin_acc[:, 2],
        torch.full((NUM_ENVS,), 9.81, dtype=lin_acc.dtype, device=lin_acc.device),
        atol=0.5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        lin_acc[:, :2],
        torch.zeros(NUM_ENVS, 2, dtype=lin_acc.dtype, device=lin_acc.device),
        atol=0.5,
        rtol=0.0,
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_freefall_acceleration(sim_ctx, device):
    """Test that a freefalling IMU measures near-zero proper acceleration.

    In freefall the finite-difference world-frame acceleration (~``-g``) cancels
    the IMU's gravity bias (``+g``), so the reading converges to ``[0, 0, 0]``.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS, height=5.0)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    # Let physics integrate gravity for a few steps with no external velocity write.
    for _ in range(10):
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        imu_ball.update(dt, force_recompute=True)

    lin_acc = imu_ball.data.lin_acc_b.torch
    acc_magnitude = torch.linalg.norm(lin_acc, dim=-1)
    torch.testing.assert_close(
        acc_magnitude,
        torch.zeros_like(acc_magnitude),
        atol=0.5,
        rtol=0.0,
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_reset(sim_ctx, device):
    """Test that ``reset`` zeroes out the IMU output and previous-velocity buffers.

    Mirrors the Newton ``test_reset`` parity check: drive the IMU until its
    buffers hold non-zero data, then ``reset()`` and assert the raw warp
    buffers are zero.  We read the raw warp arrays directly because accessing
    ``imu.data`` triggers a lazy re-fill that masks reset bugs.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    nonzero_vel = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1)
    for _ in range(5):
        balls.write_root_velocity_to_sim(nonzero_vel)
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        imu_ball.update(dt, force_recompute=True)

    # Buffers should hold non-zero gravity-bias + numerical-diff state before reset.
    assert torch.any(wp.to_torch(imu_ball._data._lin_acc_b) != 0), "expected non-zero data before reset"

    imu_ball.reset()

    # Read raw warp buffers directly — ``imu.data`` would trigger a lazy re-fill that
    # bypasses the reset.
    ang_vel_after = wp.to_torch(imu_ball._data._ang_vel_b)
    lin_acc_after = wp.to_torch(imu_ball._data._lin_acc_b)
    prev_vel_after = wp.to_torch(imu_ball._prev_lin_vel_w)
    torch.testing.assert_close(ang_vel_after, torch.zeros_like(ang_vel_after))
    torch.testing.assert_close(lin_acc_after, torch.zeros_like(lin_acc_after))
    torch.testing.assert_close(prev_vel_after, torch.zeros_like(prev_vel_after))


@pytest.mark.parametrize("device", _DEVICES)
def test_no_stale_data_after_scene_reset(sim_ctx, device):
    """Test ``scene.reset(env_ids)`` does not expose stale native velocity through ``imu.data``."""
    scene_cfg = _StaleResetSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=False)
    scene = InteractiveScene(scene_cfg)
    sim_ctx.reset()
    scene.reset()

    sensor: Imu = scene["imu_cube"]

    # Drive the native rigid-body velocity buffer non-zero. A freely falling body
    # can read zero proper acceleration, so assert the cached velocity instead.
    cube: RigidObject = scene["cube"]
    nonzero_vel = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    cube.write_root_velocity_to_sim_index(root_velocity=nonzero_vel)
    scene.write_data_to_sim()
    sim_ctx.step()
    scene.update(dt=sim_ctx.get_physics_dt())

    assert torch.any(wp.to_torch(sensor._prev_lin_vel_w) != 0), "expected non-zero cached velocity before reset"

    # Reset without another physics step. The public accessor must keep reset outputs
    # instead of lazy-refetching stale native velocity.
    scene.reset(env_ids=torch.tensor([0], device=device))

    post_reset_lin_acc = sensor.data.lin_acc_b.torch
    post_reset_ang_vel = sensor.data.ang_vel_b.torch
    torch.testing.assert_close(post_reset_lin_acc, torch.zeros_like(post_reset_lin_acc))
    torch.testing.assert_close(post_reset_ang_vel, torch.zeros_like(post_reset_ang_vel))


@pytest.mark.parametrize("device", _DEVICES)
def test_indirect_attachment_usd(sim_ctx, device):
    """Test that an IMU attached to a non-physics Xform under a rigid ancestor matches a direct attachment.

    USD-only port of PhysX's ``test_indirect_attachment``: the URDF pendulum is
    not available kitless, but the indirect-attachment code path is reachable
    by attaching a non-physics Xform child to a rigid ball and pointing the
    IMU at it.  The composed offset should match the directly-configured
    offset; ``ang_vel_b`` and ``lin_acc_b`` should agree.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    # Add a non-physics Xform child under each ball at a known offset; the IMU
    # must resolve the rigid-body ancestor (the ball) and recover the offset.
    sub_pos = (0.4, 0.0, 0.1)
    sub_rot = (0.5, 0.5, 0.5, 0.5)
    for i in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/env_{i}/ball/imu_sub", "Xform", translation=sub_pos, orientation=sub_rot)
    imu_indirect = _make_imu("/World/env_*/ball/imu_sub")
    imu_direct = _make_imu("/World/env_*/ball", offset=ImuCfg.OffsetCfg(pos=sub_pos, rot=sub_rot))
    sim_ctx.reset()

    torch.testing.assert_close(
        wp.to_torch(imu_indirect._offset_pos_b),
        wp.to_torch(imu_direct._offset_pos_b),
    )
    torch.testing.assert_close(
        wp.to_torch(imu_indirect._offset_quat_b),
        wp.to_torch(imu_direct._offset_quat_b),
        rtol=1e-4,
        atol=1e-4,
    )

    dt = sim_ctx.get_physics_dt()
    drive_vel = torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1)
    for idx in range(50):
        balls.write_root_velocity_to_sim(drive_vel * (idx + 1))
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        imu_indirect.update(dt, force_recompute=True)
        imu_direct.update(dt, force_recompute=True)

        if idx < 2:
            continue

        torch.testing.assert_close(
            imu_indirect.data.ang_vel_b.torch,
            imu_direct.data.ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            imu_indirect.data.lin_acc_b.torch,
            imu_direct.data.lin_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )


# ===========================================================================
# Validation tests (no asset state required)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
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


@pytest.mark.parametrize("device", _DEVICES)
def test_sensor_print(sim_ctx, device):
    """Test ``__str__`` is implemented and exposes the prim path and binding pattern."""
    _spawn_envs(NUM_ENVS)
    _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx.reset()

    s = str(imu_ball)
    print(s)
    assert "Imu sensor @ '/World/env_*/ball'" in s
    assert "binding pattern" in s
    assert "/World/env_*/ball" in s
    assert "number of sensors : 2" in s


# ===========================================================================
# URDF-dependent tests — skipped pending USD pendulum asset
# ===========================================================================


@pytest.mark.skip(
    reason=(
        "Blocked on a USD-converted pendulum asset (the PhysX test loads"
        " source/isaaclab_physx/test/sensors/urdfs/simple_2_link.urdf via the Kit URDF importer,"
        " which is not loaded under the direct ./isaaclab.sh -p runner). Re-enable"
        " once a pre-converted USD pendulum is available."
    )
)
def test_single_dof_pendulum():
    """Test imu against analytical pendulum problem."""


@pytest.mark.skip(
    reason=(
        "Blocked on a USD-converted pendulum asset (the PhysX test loads"
        " source/isaaclab_physx/test/sensors/urdfs/simple_2_link.urdf via the Kit URDF importer,"
        " which is not loaded under the direct ./isaaclab.sh -p runner). Re-enable"
        " once a pre-converted USD pendulum is available."
    )
)
def test_indirect_attachment():
    """Test attaching the IMU through an Xform primitive offset chain."""
