# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX PVA sensor.

Mirrors the structure of source/isaaclab_physx/test/sensors/test_pva.py
but runs kitless under ./isaaclab.sh -p -m pytest — no AppLauncher needed.
SimulationContext is instantiated directly (it does not require Kit).

Tests that load the PhysX pendulum URDF (``test_single_dof_pendulum`` and
``test_indirect_attachment``) are skipped pending a USD-converted pendulum
asset. URDF→USD conversion requires the Kit URDF importer extension, which
is not loaded under the direct ./isaaclab.sh -p runner.

All tests use procedural USD assets so the kitless suite does not depend on
Nucleus or ``omni.client`` loader state.
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

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors.pva import Pva, PvaCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

wp.init()

pytestmark = pytest.mark.device_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ENVS = 2
"""Number of environment instances spawned in each test scene."""

MOUNT_POS_OFFSET = (0.4, 0.0, 0.1)
"""Known position offset from the rigid cube to the procedural PVA child frame [m]."""

MOUNT_ROT_OFFSET = (0.5, 0.5, 0.5, 0.5)
"""Known rotation offset from the rigid cube to the procedural PVA child frame."""


# ---------------------------------------------------------------------------
# Scene-builder helpers (real backend, procedural USD assets)
# ---------------------------------------------------------------------------


def _spawn_envs(num_envs: int) -> None:
    """Create per-env Xform containers at ``/World/env_<i>``.

    These match the prim-path layout the PVA's attachment-validity test
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


def _add_pva_mount_xforms(num_envs: int, parent_name: str = "cube", child_name: str = "pva_mount") -> None:
    """Add a procedural non-physics child Xform under each rigid body."""
    for i in range(num_envs):
        sim_utils.create_prim(
            f"/World/env_{i}/{parent_name}/{child_name}",
            "Xform",
            translation=MOUNT_POS_OFFSET,
            orientation=MOUNT_ROT_OFFSET,
        )


def _make_pva(prim_path: str, offset: PvaCfg.OffsetCfg | None = None) -> Pva:
    """Create a :class:`Pva` with the given prim path and optional offset."""
    cfg = PvaCfg(prim_path=prim_path)
    if offset is not None:
        cfg.offset = offset
    return Pva(cfg)


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
    pva_cube: PvaCfg = PvaCfg(prim_path="{ENV_REGEX_NS}/cube")


# ---------------------------------------------------------------------------
# Process-global device-mode lock (matches the IMU and contact-sensor tests).
# The ovphysx wheel can only run one device per process; parametrized tests
# skip on the unlocked device so single-device runs finish cleanly.
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
        the PhysX reference test (``dt=0.001``) for PVA numerical-differentiation
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
    """Test the PVA sensor with a constant velocity.

    Expected behavior is that the linear and angular accelerations are
    approximately the same at every time step: in each step we set the same
    velocity, so the finite-difference derivative settles to zero. The PVA
    linear acceleration is the coordinate acceleration and does not include
    gravity (the IMU's gravity bias is absent here).
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    cubes = _spawn_cubes(NUM_ENVS)
    pva_ball = _make_pva("/World/env_*/ball")
    pva_cube = _make_pva("/World/env_*/cube")
    sim_ctx.reset()

    prev_lin_acc_ball = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=device)
    prev_ang_acc_ball = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=device)
    prev_lin_acc_cube = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=device)
    prev_ang_acc_cube = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=device)

    dt = sim_ctx.get_physics_dt()

    for idx in range(200):
        # set velocity
        velocity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        )
        balls.write_root_velocity_to_sim_index(root_velocity=velocity)
        cubes.write_root_velocity_to_sim_index(root_velocity=velocity)
        # write data to sim
        balls.write_data_to_sim()
        cubes.write_data_to_sim()
        # perform step
        sim_ctx.step()
        # read data from sim
        balls.update(dt)
        cubes.update(dt)
        pva_ball.update(dt, force_recompute=True)
        pva_cube.update(dt, force_recompute=True)

        if idx > 1:
            # check the pva accelerations
            torch.testing.assert_close(
                pva_ball.data.lin_acc_b.torch,
                prev_lin_acc_ball,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                pva_ball.data.ang_acc_b.torch,
                prev_ang_acc_ball,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                pva_cube.data.lin_acc_b.torch,
                prev_lin_acc_cube,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                pva_cube.data.ang_acc_b.torch,
                prev_ang_acc_cube,
                rtol=1e-3,
                atol=1e-3,
            )

            # check the pva velocities: write_root_velocity_to_sim sets v_0 every step,
            # the solver then integrates one step under gravity → v_i = v_0 + g*dt.
            expected_vel = torch.tensor([[1.0, 0.0, -dt * 9.81]], dtype=torch.float32, device=device).repeat(
                NUM_ENVS, 1
            )
            torch.testing.assert_close(
                pva_ball.data.lin_vel_b.torch,
                expected_vel,
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                pva_cube.data.lin_vel_b.torch,
                expected_vel,
                rtol=1e-4,
                atol=1e-4,
            )

        # update previous values
        prev_lin_acc_ball = pva_ball.data.lin_acc_b.torch.clone()
        prev_ang_acc_ball = pva_ball.data.ang_acc_b.torch.clone()
        prev_lin_acc_cube = pva_cube.data.lin_acc_b.torch.clone()
        prev_ang_acc_cube = pva_cube.data.ang_acc_b.torch.clone()


@pytest.mark.parametrize("device", _DEVICES)
def test_constant_acceleration(sim_ctx, device):
    """Test the PVA sensor with a constant acceleration.

    The PVA linear acceleration is a coordinate acceleration (no gravity bias),
    so it should equal the imposed dv/dt rotated into the body frame.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    pva_ball = _make_pva("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()

    for idx in range(100):
        # set acceleration via increasing velocity per step
        velocity = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        balls.write_root_velocity_to_sim_index(root_velocity=velocity)
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        pva_ball.update(dt, force_recompute=True)

        # skip first step where initial velocity is zero
        if idx < 1:
            continue

        # PVA linear acceleration (coordinate) — no gravity bias.
        torch.testing.assert_close(
            pva_ball.data.lin_acc_b.torch,
            math_utils.quat_apply_inverse(
                balls.data.root_quat_w.torch,
                torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1) / dt,
            ),
            rtol=1e-4,
            atol=1e-4,
        )

        # check the angular velocity
        torch.testing.assert_close(
            pva_ball.data.ang_vel_b.torch,
            balls.data.root_ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )


# ===========================================================================
# Offset and env-id tests (procedural USD assets)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
def test_offset_calculation(sim_ctx, device):
    """Test offset configuration argument.

    Two PVA sensors on the same cube -- one attached to a non-physics child
    Xform and one attached to the cube with the same configured offset -- should
    produce identical readings across all outputs.
    """
    _spawn_envs(NUM_ENVS)
    cubes = _spawn_cubes(NUM_ENVS)
    _add_pva_mount_xforms(NUM_ENVS)
    pva_child = _make_pva("/World/env_*/cube/pva_mount")
    pva_direct = _make_pva(
        "/World/env_*/cube",
        offset=PvaCfg.OffsetCfg(pos=MOUNT_POS_OFFSET, rot=MOUNT_ROT_OFFSET),
    )
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()

    for idx in range(50):
        velocity = torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        cubes.write_root_velocity_to_sim_index(root_velocity=velocity)
        cubes.write_data_to_sim()
        sim_ctx.step()
        cubes.update(dt)
        pva_child.update(dt, force_recompute=True)
        pva_direct.update(dt, force_recompute=True)

        # skip first step where initial velocity is zero
        if idx < 1:
            continue

        # accelerations
        torch.testing.assert_close(
            pva_direct.data.lin_acc_b.torch,
            pva_child.data.lin_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_direct.data.ang_acc_b.torch,
            pva_child.data.ang_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # velocities
        torch.testing.assert_close(
            pva_direct.data.lin_vel_b.torch,
            pva_child.data.lin_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_direct.data.ang_vel_b.torch,
            pva_child.data.ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # pose
        torch.testing.assert_close(
            pva_direct.data.pos_w.torch,
            pva_child.data.pos_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_direct.data.quat_w.torch,
            pva_child.data.quat_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # projected gravity
        torch.testing.assert_close(
            pva_direct.data.projected_gravity_b.torch,
            pva_child.data.projected_gravity_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_env_ids_propagation(sim_ctx, device):
    """Test that ``env_ids`` argument propagates through update and reset methods."""
    _spawn_envs(NUM_ENVS)
    cubes = _spawn_cubes(NUM_ENVS)
    pva_cube = _make_pva("/World/env_*/cube")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()

    for idx in range(10):
        velocity = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        cubes.write_root_velocity_to_sim_index(root_velocity=velocity)
        cubes.write_data_to_sim()
        sim_ctx.step()
        cubes.update(dt)
        pva_cube.update(dt, force_recompute=True)

    assert torch.any(pva_cube.data.lin_vel_b.torch[1] != 0), "expected env 1 to have non-zero data before reset"

    # reset only env 1
    pva_cube.reset(env_ids=[1])
    torch.testing.assert_close(
        wp.to_torch(pva_cube._data._lin_vel_b)[1],
        torch.zeros(3, dtype=torch.float32, device=device),
    )
    assert torch.any(wp.to_torch(pva_cube._data._lin_vel_b)[0] != 0), "env 0 should not be reset"

    pva_cube.update(dt, force_recompute=True)
    sim_ctx.step()
    pva_cube.update(dt, force_recompute=True)


# ===========================================================================
# Physics-correctness sanity tests (ported from Newton)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
def test_sensor_initialization(sim_ctx, device):
    """Test that the OVPhysX PVA sensor initializes correctly."""
    _spawn_envs(NUM_ENVS)
    _spawn_balls(NUM_ENVS)
    pva_ball = _make_pva("/World/env_*/ball")
    sim_ctx.reset()

    assert pva_ball.num_instances == NUM_ENVS
    # Inspect the raw warp buffers directly — accessing ``pva.data`` triggers a
    # lazy FD-acceleration recompute that needs ``_dt`` (set by ``update``).
    for name, expected_dtype in [
        ("_pose_w", wp.transformf),
        ("_pos_w", wp.vec3f),
        ("_quat_w", wp.quatf),
        ("_lin_vel_b", wp.vec3f),
        ("_ang_vel_b", wp.vec3f),
        ("_lin_acc_b", wp.vec3f),
        ("_ang_acc_b", wp.vec3f),
        ("_projected_gravity_b", wp.vec3f),
    ]:
        buf = getattr(pva_ball._data, name)
        assert buf.shape == (NUM_ENVS,), f"{name} shape mismatch"
        assert buf.dtype == expected_dtype, f"{name} dtype mismatch"


@pytest.mark.parametrize("device", _DEVICES)
def test_pose_w_packing(sim_ctx, device):
    """Verify the lazy ``pose_w`` property packs ``[pos_w | quat_w]`` correctly.

    Regression guard for the ``concat_pos_and_quat_to_pose_1d_kernel`` launch in
    :meth:`PvaData.pose_w` — without this test only buffer shape/dtype is
    checked, and a regression that, say, swaps the kernel inputs or returns
    stale data would not be caught.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    pva_ball = _make_pva("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    # Step a few times so pos_w / quat_w are populated by the simulation.
    for _ in range(3):
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        pva_ball.update(dt, force_recompute=True)

    pose = pva_ball.data.pose_w.torch
    assert pose.shape == (NUM_ENVS, 7)
    torch.testing.assert_close(pose[:, :3], pva_ball.data.pos_w.torch)
    torch.testing.assert_close(pose[:, 3:], pva_ball.data.quat_w.torch)


@pytest.mark.parametrize("device", _DEVICES)
def test_projected_gravity_at_rest(sim_ctx, device):
    """Test that a PVA at rest reports projected gravity ≈ (0, 0, -1) in body frame.

    Without InteractiveScene's terrain plumbing the ball falls forever, so we
    drive it kinematically: hold zero velocity for enough steps to settle, then
    check that the projected gravity unit vector points along world ``-z`` —
    which for a body whose orientation is identity is also ``(0, 0, -1)`` in
    the body frame.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    pva_ball = _make_pva("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    zero_vel = torch.zeros((NUM_ENVS, 6), dtype=torch.float32, device=device)
    for _ in range(5):
        balls.write_root_velocity_to_sim_index(root_velocity=zero_vel)
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        pva_ball.update(dt, force_recompute=True)

    pg = pva_ball.data.projected_gravity_b.torch
    expected = torch.tensor([[0.0, 0.0, -1.0]], dtype=pg.dtype, device=pg.device).repeat(NUM_ENVS, 1)
    torch.testing.assert_close(pg, expected, atol=1e-3, rtol=0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_freefall_lin_acc(sim_ctx, device):
    """Test that a freefalling PVA reports lin_acc_b ≈ rotated ``-g`` in body frame.

    The PVA reports the coordinate acceleration of the sensor frame. In
    freefall this is ``-g`` (the body is accelerating downward at ``g``), so
    the magnitude of ``lin_acc_b`` should converge to ``g ≈ 9.81 m/s^2``.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS, height=5.0)
    pva_ball = _make_pva("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    # Let physics integrate gravity for a few steps with no external velocity write.
    for _ in range(10):
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        pva_ball.update(dt, force_recompute=True)

    lin_acc = pva_ball.data.lin_acc_b.torch
    acc_magnitude = torch.linalg.norm(lin_acc, dim=-1)
    torch.testing.assert_close(
        acc_magnitude,
        torch.full((NUM_ENVS,), 9.81, dtype=acc_magnitude.dtype, device=acc_magnitude.device),
        atol=0.5,
        rtol=0.0,
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_reset(sim_ctx, device):
    """Test that ``reset`` zeroes out the PVA output and previous-velocity buffers.

    Mirrors the Newton ``test_reset`` parity check: drive the PVA until its
    buffers hold non-zero data, then ``reset()`` and assert the raw warp
    buffers are zero.  We read the raw warp arrays directly because accessing
    ``pva.data`` triggers a lazy re-fill that masks reset bugs.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    pva_ball = _make_pva("/World/env_*/ball")
    sim_ctx.reset()

    dt = sim_ctx.get_physics_dt()
    nonzero_vel = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1)
    for _ in range(5):
        balls.write_root_velocity_to_sim_index(root_velocity=nonzero_vel)
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        pva_ball.update(dt, force_recompute=True)

    # Buffers should hold non-zero state before reset.
    assert torch.any(wp.to_torch(pva_ball._data._lin_vel_b) != 0), "expected non-zero data before reset"

    pva_ball.reset()

    # Read raw warp buffers directly — ``pva.data`` would trigger a lazy re-fill that
    # bypasses the reset.
    for name in (
        "_pos_w",
        "_lin_vel_b",
        "_ang_vel_b",
        "_lin_acc_b",
        "_ang_acc_b",
    ):
        buf = wp.to_torch(getattr(pva_ball._data, name))
        torch.testing.assert_close(buf, torch.zeros_like(buf), msg=f"{name} not zeroed by reset()")

    # quat_w resets to identity (x, y, z, w) = (0, 0, 0, 1)
    quat = wp.to_torch(pva_ball._data._quat_w)
    expected_quat = torch.zeros_like(quat)
    expected_quat[:, 3] = 1.0
    torch.testing.assert_close(quat, expected_quat)

    # projected_gravity_b resets to (0, 0, -1)
    pg = wp.to_torch(pva_ball._data._projected_gravity_b)
    expected_pg = torch.zeros_like(pg)
    expected_pg[:, 2] = -1.0
    torch.testing.assert_close(pg, expected_pg)

    # previous-velocity buffers cleared
    torch.testing.assert_close(
        wp.to_torch(pva_ball._prev_lin_vel_w), torch.zeros_like(wp.to_torch(pva_ball._prev_lin_vel_w))
    )
    torch.testing.assert_close(
        wp.to_torch(pva_ball._prev_ang_vel_w), torch.zeros_like(wp.to_torch(pva_ball._prev_ang_vel_w))
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_no_stale_data_after_scene_reset(sim_ctx, device):
    """Test ``scene.reset(env_ids)`` does not expose stale native velocity through ``pva.data``."""
    scene_cfg = _StaleResetSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=False)
    scene = InteractiveScene(scene_cfg)
    sim_ctx.reset()
    scene.reset()

    sensor: Pva = scene["pva_cube"]

    # Drive the native rigid-body velocity buffer non-zero. Freefall can make
    # acceleration assertions depend on the exact step, so assert the cached
    # finite-difference state instead.
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
    post_reset_ang_acc = sensor.data.ang_acc_b.torch
    post_reset_lin_vel = sensor.data.lin_vel_b.torch
    post_reset_ang_vel = sensor.data.ang_vel_b.torch
    torch.testing.assert_close(post_reset_lin_acc, torch.zeros_like(post_reset_lin_acc))
    torch.testing.assert_close(post_reset_ang_acc, torch.zeros_like(post_reset_ang_acc))
    torch.testing.assert_close(post_reset_lin_vel, torch.zeros_like(post_reset_lin_vel))
    torch.testing.assert_close(post_reset_ang_vel, torch.zeros_like(post_reset_ang_vel))


@pytest.mark.parametrize("device", _DEVICES)
def test_indirect_attachment_usd(sim_ctx, device):
    """Test that a PVA attached to a non-physics Xform under a rigid ancestor matches a direct attachment.

    USD-only port of PhysX's ``test_indirect_attachment``: the URDF pendulum is
    not available kitless, but the indirect-attachment code path is reachable
    by attaching a non-physics Xform child to a rigid ball and pointing the
    PVA at it.  The composed offset should match the directly-configured
    offset; all output channels should agree.
    """
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    # Add a non-physics Xform child under each ball at a known offset; the PVA
    # must resolve the rigid-body ancestor (the ball) and recover the offset.
    sub_pos = (0.4, 0.0, 0.1)
    sub_rot = (0.5, 0.5, 0.5, 0.5)
    for i in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/env_{i}/ball/pva_sub", "Xform", translation=sub_pos, orientation=sub_rot)
    pva_indirect = _make_pva("/World/env_*/ball/pva_sub")
    pva_direct = _make_pva("/World/env_*/ball", offset=PvaCfg.OffsetCfg(pos=sub_pos, rot=sub_rot))
    sim_ctx.reset()

    torch.testing.assert_close(
        wp.to_torch(pva_indirect._offset_pos_b),
        wp.to_torch(pva_direct._offset_pos_b),
    )
    torch.testing.assert_close(
        wp.to_torch(pva_indirect._offset_quat_b),
        wp.to_torch(pva_direct._offset_quat_b),
        rtol=1e-4,
        atol=1e-4,
    )

    dt = sim_ctx.get_physics_dt()
    drive_vel = torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(NUM_ENVS, 1)
    for idx in range(50):
        balls.write_root_velocity_to_sim_index(root_velocity=drive_vel * (idx + 1))
        balls.write_data_to_sim()
        sim_ctx.step()
        balls.update(dt)
        pva_indirect.update(dt, force_recompute=True)
        pva_direct.update(dt, force_recompute=True)

        if idx < 2:
            continue

        torch.testing.assert_close(
            pva_indirect.data.pos_w.torch,
            pva_direct.data.pos_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_indirect.data.quat_w.torch,
            pva_direct.data.quat_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_indirect.data.lin_vel_b.torch,
            pva_direct.data.lin_vel_b.torch,
            rtol=1e-2,
            atol=5e-3,
        )
        torch.testing.assert_close(
            pva_indirect.data.ang_vel_b.torch,
            pva_direct.data.ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_indirect.data.lin_acc_b.torch,
            pva_direct.data.lin_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_indirect.data.ang_acc_b.torch,
            pva_direct.data.ang_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pva_indirect.data.projected_gravity_b.torch,
            pva_direct.data.projected_gravity_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )


# ===========================================================================
# Validation tests (no asset state required)
# ===========================================================================


@pytest.mark.parametrize("device", _DEVICES)
def test_attachment_validity(sim_ctx, device):
    """Test invalid PVA attachment.

    A PVA sensor cannot be attached directly to the world Xform — it must have
    a rigid-body ancestor in its prim tree.
    """
    _spawn_envs(NUM_ENVS)
    sim_ctx.reset()

    pva_world_cfg = PvaCfg(prim_path="/World/env_0")
    with pytest.raises(RuntimeError) as exc_info:
        pva_world = Pva(pva_world_cfg)
        pva_world._initialize_impl()
    assert exc_info.type is RuntimeError and "find a rigid body ancestor prim" in str(exc_info.value)


@pytest.mark.parametrize("device", _DEVICES)
def test_sensor_print(sim_ctx, device):
    """Test ``__str__`` is implemented and exposes the prim path and binding pattern."""
    _spawn_envs(NUM_ENVS)
    _spawn_balls(NUM_ENVS)
    pva_ball = _make_pva("/World/env_*/ball")
    sim_ctx.reset()

    s = str(pva_ball)
    print(s)
    assert "Pva sensor @ '/World/env_*/ball'" in s
    assert "binding pattern" in s
    assert "/World/env_*/ball" in s
    assert "number of sensors : 2" in s


# ===========================================================================
# URDF-dependent tests — skipped pending USD pendulum asset
# ===========================================================================


_PENDULUM_SKIP_REASON = (
    "Blocked on a USD-converted pendulum asset (the PhysX test loads"
    " source/isaaclab_physx/test/sensors/urdfs/simple_2_link.urdf via the Kit URDF importer,"
    " which is not loaded under the direct ./isaaclab.sh -p runner). Re-enable"
    " once a pre-converted USD pendulum is available, and port the assertion body from"
    " source/isaaclab_physx/test/sensors/test_pva.py."
)


@pytest.mark.skip(reason=_PENDULUM_SKIP_REASON)
def test_single_dof_pendulum():
    """Test PVA against analytical pendulum problem."""
    # If this test is ever un-skipped without porting the PhysX assertions, fail
    # explicitly rather than passing vacuously.
    pytest.fail(
        "test_single_dof_pendulum was un-skipped without a body — port the assertions from"
        " source/isaaclab_physx/test/sensors/test_pva.py::test_single_dof_pendulum and adapt"
        " them to the kitless RigidObject/Articulation pattern used by the rest of this file."
    )


@pytest.mark.skip(reason=_PENDULUM_SKIP_REASON)
def test_indirect_attachment():
    """Test attaching the PVA through an Xform primitive offset chain."""
    # If this test is ever un-skipped without porting the PhysX assertions, fail
    # explicitly rather than passing vacuously.
    pytest.fail(
        "test_indirect_attachment was un-skipped without a body — port the assertions from"
        " source/isaaclab_physx/test/sensors/test_pva.py::test_indirect_attachment and adapt"
        " them to the kitless RigidObject/Articulation pattern used by the rest of this file."
    )
