# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests comparing WrenchComposer output vs raw PhysX apply_forces_and_torques_at_position.

Two identical rigid objects are placed in the same scene. One uses the WrenchComposer path
(set_forces_and_torques → write_data_to_sim → compose → PhysX apply with is_global=False),
the other uses the raw PhysX API directly (apply_forces_and_torques_at_position with matching
is_global flag). After N steps, both objects should have identical velocities.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def generate_dual_cube_scene(
    num_cubes: int = 1,
    height: float = 1.0,
    device: str = "cuda:0",
    initial_rot: tuple[float, ...] | None = None,
    spacing: float = 2.0,
) -> tuple[RigidObject, RigidObject]:
    """Generate a scene with two sets of cubes: one for the composer path, one for raw PhysX.

    Both sets share the same spawn config and initial state (except a Y offset to avoid overlap).

    Args:
        num_cubes: Number of cubes per group (environments).
        height: Spawn height.
        device: Simulation device.
        initial_rot: Initial quaternion (x, y, z, w). Defaults to identity.
        spacing: Distance between env origins in X. Defaults to 2.0.

    Returns:
        Tuple of (cube_composer, cube_raw) RigidObject instances.
    """
    if initial_rot is None:
        initial_rot = (0.0, 0.0, 0.0, 1.0)  # identity in (x,y,z,w)

    y_offset = max(spacing, 3.0)

    # Create Xform prims for both groups
    for i in range(num_cubes):
        origin_composer = (i * spacing, 0.0, height)
        origin_raw = (i * spacing, y_offset, height)  # Y offset to avoid overlap
        sim_utils.create_prim(f"/World/Composer_{i}", "Xform", translation=origin_composer)
        sim_utils.create_prim(f"/World/Raw_{i}", "Xform", translation=origin_raw)

    spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )

    cube_composer_cfg = RigidObjectCfg(
        prim_path="/World/Composer_.*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height), rot=initial_rot),
    )
    cube_composer = RigidObject(cfg=cube_composer_cfg)

    cube_raw_cfg = RigidObjectCfg(
        prim_path="/World/Raw_.*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, y_offset, height), rot=initial_rot),
    )
    cube_raw = RigidObject(cfg=cube_raw_cfg)

    return cube_composer, cube_raw


N_STEPS = 50
FORCE_MAGNITUDE = 10.0
TORQUE_MAGNITUDE = 1.0
# 45 degrees about Z: (cos(22.5°), 0, 0, sin(22.5°))
ROT_45_Z = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))  # 45deg about Z in (x,y,z,w)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_local_force(device):
    """Baseline: local force at identity orientation. Composer and raw PhysX should match exactly."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Composer path: local force +X
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=False,
        )

        # Raw PhysX data (flattened for PhysX view API)
        raw_forces = torch.zeros(1, 3, device=device)
        raw_forces[:, 0] = FORCE_MAGNITUDE
        raw_torques = torch.zeros(1, 3, device=device)
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()  # no-op (composer inactive)
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=False,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Compare velocities
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # Both should have ~zero angular velocity (force at CoM, no torque)
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            torch.zeros(1, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )
        torch.testing.assert_close(
            cube_raw.data.root_ang_vel_w.torch,
            torch.zeros(1, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_global_force(device):
    """Global force with non-identity rotation (45 deg Z). Rotation matters for frame conversion."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(num_cubes=1, device=device, initial_rot=ROT_45_Z)

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Composer path: global force +X
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=True,
        )

        # Raw PhysX data
        raw_forces = torch.zeros(1, 3, device=device)
        raw_forces[:, 0] = FORCE_MAGNITUDE
        raw_torques = torch.zeros(1, 3, device=device)
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Linear velocities should match (same global force, same mass)
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # Angular velocities should match
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # Both should have ~zero angular velocity (force at CoM, no torque)
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            torch.zeros(1, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )
        torch.testing.assert_close(
            cube_raw.data.root_ang_vel_w.torch,
            torch.zeros(1, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_local_force_at_position(device):
    """Local force at a local offset. Both paths should produce identical cross-product torque."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Local force +X at local offset +0.5m Y
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)
        positions = torch.zeros(1, len(body_ids), 3, device=device)
        positions[..., 1] = 0.5  # +0.5m Y offset in local frame

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            is_global=False,
        )

        # Raw PhysX data (local force at local position)
        raw_forces = torch.zeros(1, 3, device=device)
        raw_forces[:, 0] = FORCE_MAGNITUDE
        raw_torques = torch.zeros(1, 3, device=device)
        raw_positions = torch.zeros(1, 3, device=device)
        raw_positions[:, 1] = 0.5
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=wp.from_torch(raw_positions.contiguous(), dtype=wp.float32),
                indices=raw_indices,
                is_global=False,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Both linear and angular velocities should match
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )

        # Sanity: angular velocity should be nonzero (cross-product torque)
        assert torch.abs(cube_composer.data.root_ang_vel_w.torch[0, 2]).item() > 0.1, (
            "Expected nonzero Z angular velocity from cross-product torque"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_global_force_at_position(device):
    """Global force at world position with non-identity rotation. Both rotation AND position correction matter."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(num_cubes=1, device=device, initial_rot=ROT_45_Z)

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Global force +X
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        # Position = each cube's link_pos + offset (same offset for both)
        offset = torch.zeros(1, len(body_ids), 3, device=device)
        offset[..., 1] = 1.0  # +1m Y offset in world frame

        pos_composer = cube_composer.data.body_com_pos_w.torch[:, body_ids, :3].clone() + offset
        pos_raw = cube_raw.data.body_com_pos_w.torch[:, body_ids, :3].clone() + offset

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=pos_composer,
            body_ids=body_ids,
            is_global=True,
        )

        # Raw PhysX data
        raw_forces = torch.zeros(1, 3, device=device)
        raw_forces[:, 0] = FORCE_MAGNITUDE
        raw_torques = torch.zeros(1, 3, device=device)
        raw_positions = pos_raw.view(-1, 3)
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=wp.from_torch(raw_positions.contiguous(), dtype=wp.float32),
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Both linear and angular velocities should match
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )

        # Sanity: angular velocity should be nonzero (cross-product torque)
        assert torch.abs(cube_composer.data.root_ang_vel_w.torch[0, 2]).item() > 0.1, (
            "Expected nonzero Z angular velocity from positional torque"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_local_torque(device):
    """Local torque at identity orientation. Should produce matching angular velocity."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Composer path: local torque about +Z
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        torques = torch.zeros(1, len(body_ids), 3, device=device)
        torques[..., 2] = TORQUE_MAGNITUDE

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=False,
        )

        # Raw PhysX data
        raw_forces = torch.zeros(1, 3, device=device)
        raw_torques = torch.zeros(1, 3, device=device)
        raw_torques[:, 2] = TORQUE_MAGNITUDE
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=False,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Angular velocities should match
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # Linear velocity should be ~zero for both (no force)
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            torch.zeros(1, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )
        torch.testing.assert_close(
            cube_raw.data.root_lin_vel_w.torch,
            torch.zeros(1, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_global_torque(device):
    """Global torque with non-identity rotation (45 deg Z). Composer rotates to body frame internally."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(num_cubes=1, device=device, initial_rot=ROT_45_Z)

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Composer path: global torque about +Z
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        torques = torch.zeros(1, len(body_ids), 3, device=device)
        torques[..., 2] = TORQUE_MAGNITUDE

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=True,
        )

        # Raw PhysX data
        raw_forces = torch.zeros(1, 3, device=device)
        raw_torques = torch.zeros(1, 3, device=device)
        raw_torques[:, 2] = TORQUE_MAGNITUDE
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Angular velocities should match
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )


NUM_CUBES_MULTI = 4


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_global_force_multi_env(device):
    """Global force (no position) with multiple environments.

    Regression: checks that env-indexing and per-body quaternion handling work correctly
    when there is more than one environment.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(
            num_cubes=NUM_CUBES_MULTI, device=device, initial_rot=ROT_45_Z
        )

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Composer path: global force +X for all envs
        forces = torch.zeros(NUM_CUBES_MULTI, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(NUM_CUBES_MULTI, len(body_ids), 3, device=device)

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=True,
        )

        # Raw PhysX data (one row per env)
        raw_forces = torch.zeros(NUM_CUBES_MULTI, 3, device=device)
        raw_forces[:, 0] = FORCE_MAGNITUDE
        raw_torques = torch.zeros(NUM_CUBES_MULTI, 3, device=device)
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Linear velocities should match across all envs
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # Angular velocities should match
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # All envs should have ~zero angular velocity
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            torch.zeros(NUM_CUBES_MULTI, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_global_force_with_reset(device):
    """Global force (no position) with a mid-simulation reset of half the envs.

    Regression: after reset the permanent wrench is cleared. Re-setting it should
    produce correct behavior even though the object state was just reset.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(
            num_cubes=NUM_CUBES_MULTI, device=device, initial_rot=ROT_45_Z, spacing=20.0
        )

        sim.reset()

        # Capture initial world-frame state (includes env origin offsets)
        cube_composer.update(sim.cfg.dt)
        cube_raw.update(sim.cfg.dt)
        initial_state_composer = torch.cat(
            [
                cube_composer.data.root_link_pos_w.torch,
                cube_composer.data.root_link_quat_w.torch,
                cube_composer.data.root_com_vel_w.torch,
            ],
            dim=-1,
        ).clone()
        initial_state_raw = torch.cat(
            [
                cube_raw.data.root_link_pos_w.torch,
                cube_raw.data.root_link_quat_w.torch,
                cube_raw.data.root_com_vel_w.torch,
            ],
            dim=-1,
        ).clone()

        body_ids, _ = cube_composer.find_bodies(".*")

        def apply_global_force():
            """Set the same global +X force on the composer cube."""
            forces = torch.zeros(NUM_CUBES_MULTI, len(body_ids), 3, device=device)
            forces[..., 0] = FORCE_MAGNITUDE
            torques = torch.zeros(NUM_CUBES_MULTI, len(body_ids), 3, device=device)
            cube_composer.permanent_wrench_composer.set_forces_and_torques(
                forces=forces,
                torques=torques,
                body_ids=body_ids,
                is_global=True,
            )

        apply_global_force()

        # Raw PhysX data
        raw_forces = torch.zeros(NUM_CUBES_MULTI, 3, device=device)
        raw_forces[:, 0] = FORCE_MAGNITUDE
        raw_torques = torch.zeros(NUM_CUBES_MULTI, 3, device=device)
        raw_indices = cube_raw._ALL_INDICES

        # Phase 1: run N_STEPS / 2
        half = N_STEPS // 2
        for _ in range(half):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Reset first half of envs on both cubes
        reset_ids = list(range(NUM_CUBES_MULTI // 2))
        reset_ids_torch = torch.tensor(reset_ids, dtype=torch.long, device=device)

        # Reset root state using captured world-frame initial state (includes env origins)
        cube_composer.write_root_state_to_sim(initial_state_composer[reset_ids_torch], env_ids=reset_ids_torch)
        cube_raw.write_root_state_to_sim(initial_state_raw[reset_ids_torch], env_ids=reset_ids_torch)

        cube_composer.reset(reset_ids)
        cube_raw.reset(reset_ids)

        # Re-apply the force (reset cleared the permanent wrench)
        apply_global_force()

        # Phase 2: run N_STEPS / 2 more
        for _ in range(half):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # All envs: composer vs raw should match
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # All envs should have ~zero angular velocity
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            torch.zeros(NUM_CUBES_MULTI, 3, device=device),
            rtol=0.0,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_payload_scenario(device):
    """Mirrors the apply_payload MDP: permanent global downward force at CoM with gravity.

    A constant world-frame downward force (payload weight) is applied via the composer
    path vs raw PhysX. The body falls under gravity + payload, contacts the ground, and
    orientation changes. The composer does a world->body->world round-trip each step;
    this test catches any precision drift from that.
    """
    with build_simulation_context(device=device, gravity_enabled=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(
            num_cubes=1, height=0.5, device=device, initial_rot=ROT_45_Z, spacing=20.0
        )

        sim.reset()
        cube_composer.update(sim.cfg.dt)
        cube_raw.update(sim.cfg.dt)

        # Record initial positions to compare displacements (cubes spawn at different Y)
        init_pos_composer = cube_composer.data.root_pos_w.torch.clone()
        init_pos_raw = cube_raw.data.root_pos_w.torch.clone()

        body_ids, _ = cube_composer.find_bodies(".*")

        payload_force = 2.0 * 9.81
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 2] = -payload_force
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=True,
        )

        raw_forces = torch.zeros(1, 3, device=device)
        raw_forces[:, 2] = -payload_force
        raw_torques = torch.zeros(1, 3, device=device)
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(N_STEPS):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=None,
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        # Compare displacements (not absolute positions — cubes have different spawn Y)
        disp_composer = cube_composer.data.root_pos_w.torch - init_pos_composer
        disp_raw = cube_raw.data.root_pos_w.torch - init_pos_raw

        torch.testing.assert_close(disp_composer, disp_raw, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_vs_physx_permanent_global_force_at_position_long_run(device):
    """Permanent global force at a world-frame offset, run long enough for significant body motion.

    This test catches temporal drift bugs where the stored positional torque diverges from
    what PhysX computes each step as the body moves. The force is large enough that the body
    translates and rotates significantly over 100 steps, but not so large that it causes
    numerical instability.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_composer, cube_raw = generate_dual_cube_scene(num_cubes=1, device=device, initial_rot=ROT_45_Z)

        sim.reset()

        body_ids, _ = cube_composer.find_bodies(".*")

        # Global force +Z at +1m Y offset from CoM — produces torque around X
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 2] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        offset = torch.zeros(1, len(body_ids), 3, device=device)
        offset[..., 1] = 1.0

        pos_composer = cube_composer.data.body_com_pos_w.torch[:, body_ids, :3].clone() + offset
        pos_raw = cube_raw.data.body_com_pos_w.torch[:, body_ids, :3].clone() + offset

        cube_composer.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=pos_composer,
            body_ids=body_ids,
            is_global=True,
        )

        raw_forces = torch.zeros(1, 3, device=device)
        raw_forces[:, 2] = FORCE_MAGNITUDE
        raw_torques = torch.zeros(1, 3, device=device)
        raw_positions = pos_raw.view(-1, 3)
        raw_indices = cube_raw._ALL_INDICES

        for _ in range(100):
            cube_composer.write_data_to_sim()
            cube_raw.write_data_to_sim()
            cube_raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(raw_forces.contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(raw_torques.contiguous(), dtype=wp.float32),
                position_data=wp.from_torch(raw_positions.contiguous(), dtype=wp.float32),
                indices=raw_indices,
                is_global=True,
            )
            sim.step()
            cube_composer.update(sim.cfg.dt)
            cube_raw.update(sim.cfg.dt)

        torch.testing.assert_close(
            cube_composer.data.root_lin_vel_w.torch,
            cube_raw.data.root_lin_vel_w.torch,
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(
            cube_composer.data.root_ang_vel_w.torch,
            cube_raw.data.root_ang_vel_w.torch,
            rtol=1e-3,
            atol=1e-3,
        )

        # Sanity: angular velocity should be nonzero
        assert torch.abs(cube_composer.data.root_ang_vel_w.torch).max().item() > 0.1, (
            "Expected nonzero angular velocity from positional torque over 100 steps"
        )
