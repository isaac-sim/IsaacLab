# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX IMU sensor.

Mirrors the structure of source/isaaclab_physx/test/sensors/test_imu.py
but runs kitless under ./scripts/run_ovphysx.sh — no AppLauncher needed.
SimulationContext is instantiated directly (it does not require Kit), and
UsdFileCfg(usd_path=ISAAC_NUCLEUS_DIR/...) downloads Nucleus assets via
omni.client (which works standalone in Kit's Python).

Tests that depend on URDF→USD conversion are skipped pending a USD
version of source/isaaclab_ovphysx/test/sensors/urdfs/simple_2_link.urdf.
The URDF importer is a Kit extension and is not loaded under the
kitless launcher.
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
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors.imu import Imu, ImuCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa: E402

wp.init()

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


# ---------------------------------------------------------------------------
# Sim context fixture (real OVPhysX backend, CPU)
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_ctx_cpu():
    """Build an OVPhysX-backed :class:`SimulationContext` on CPU.

    Yields:
        The simulation context, set up with a small fixed timestep matching
        the PhysX reference test (``dt=0.001``) for IMU numerical-differentiation
        accuracy.
    """
    with build_simulation_context(
        sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device="cpu", dt=0.001),
    ) as sim:
        yield sim


# ===========================================================================
# Constant-velocity / constant-acceleration tests (rigid bodies)
# ===========================================================================


def test_constant_velocity(sim_ctx_cpu):
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
    sim_ctx_cpu.reset()

    device = sim_ctx_cpu.device
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
        sim_ctx_cpu.step()
        # read data from sim
        dt = sim_ctx_cpu.get_physics_dt()
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


def test_constant_acceleration(sim_ctx_cpu):
    """Test the IMU sensor with a constant acceleration."""
    _spawn_envs(NUM_ENVS)
    balls = _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx_cpu.reset()

    device = sim_ctx_cpu.device
    dt = sim_ctx_cpu.get_physics_dt()

    for idx in range(100):
        # set acceleration via increasing velocity per step
        velocity = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        balls.write_root_velocity_to_sim(velocity)
        balls.write_data_to_sim()
        sim_ctx_cpu.step()
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


def test_offset_calculation(sim_ctx_cpu):
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
    sim_ctx_cpu.reset()

    device = sim_ctx_cpu.device
    dt = sim_ctx_cpu.get_physics_dt()

    for idx in range(500):
        # apply increasing root velocity
        velocity = torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        robot.write_root_velocity_to_sim(velocity)
        robot.write_data_to_sim()
        sim_ctx_cpu.step()
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


def test_env_ids_propagation(sim_ctx_cpu):
    """Test that ``env_ids`` argument propagates through update and reset methods."""
    _spawn_envs(NUM_ENVS)
    robot = _spawn_anymal(NUM_ENVS)
    imu_robot_imu_link = _make_imu("/World/env_*/robot/base/imu_link")
    sim_ctx_cpu.reset()

    device = sim_ctx_cpu.device
    dt = sim_ctx_cpu.get_physics_dt()

    for idx in range(10):
        velocity = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(
            NUM_ENVS, 1
        ) * (idx + 1)
        robot.write_root_velocity_to_sim(velocity)
        robot.write_data_to_sim()
        sim_ctx_cpu.step()
        robot.update(dt)
        imu_robot_imu_link.update(dt, force_recompute=True)

    # reset only env 1
    imu_robot_imu_link.reset(env_ids=[1])
    imu_robot_imu_link.update(dt, force_recompute=True)
    sim_ctx_cpu.step()
    imu_robot_imu_link.update(dt, force_recompute=True)


# ===========================================================================
# Validation tests (no asset state required)
# ===========================================================================


def test_attachment_validity(sim_ctx_cpu):
    """Test invalid IMU attachment.

    An IMU cannot be attached directly to the world Xform — it must have a
    rigid-body ancestor in its prim tree.
    """
    _spawn_envs(NUM_ENVS)
    sim_ctx_cpu.reset()

    imu_world_cfg = ImuCfg(prim_path="/World/env_0")
    with pytest.raises(RuntimeError) as exc_info:
        imu_world = Imu(imu_world_cfg)
        imu_world._initialize_impl()
    assert exc_info.type is RuntimeError and "find a rigid body ancestor prim" in str(exc_info.value)


def test_sensor_print(sim_ctx_cpu):
    """Test that ``__str__`` is implemented and runs without error."""
    _spawn_envs(NUM_ENVS)
    _spawn_balls(NUM_ENVS)
    imu_ball = _make_imu("/World/env_*/ball")
    sim_ctx_cpu.reset()

    print(imu_ball)


# ===========================================================================
# URDF-dependent tests — skipped pending USD pendulum asset
# ===========================================================================


@pytest.mark.skip(
    reason=(
        "Blocked on USD version of source/isaaclab_ovphysx/test/sensors/urdfs/simple_2_link.urdf."
        " URDF→USD conversion requires the Kit URDF importer extension, which is not loaded under"
        " the kitless ./scripts/run_ovphysx.sh launcher. Re-enable once a pre-converted USD"
        " pendulum asset is available."
    )
)
def test_single_dof_pendulum():
    """Test imu against analytical pendulum problem."""


@pytest.mark.skip(
    reason=(
        "Blocked on USD version of source/isaaclab_ovphysx/test/sensors/urdfs/simple_2_link.urdf."
        " URDF→USD conversion requires the Kit URDF importer extension, which is not loaded under"
        " the kitless ./scripts/run_ovphysx.sh launcher. Re-enable once a pre-converted USD"
        " pendulum asset is available."
    )
)
def test_indirect_attachment():
    """Test attaching the IMU through an Xform primitive offset chain."""
