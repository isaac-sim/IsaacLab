# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for wrench composer with rigid objects.

These tests validate that global forces/torques remain invariant under body rotation
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def generate_cubes_scene(
    num_cubes: int = 1,
    height: float = 1.0,
    device: str = "cuda:0",
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes."""
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )

    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)
    return cube_object, origins


N_STEPS = 100
FORCE_MAGNITUDE = 10.0
TORQUE_MAGNITUDE = 1.0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_invariant_under_rotation(device):
    """Test that a permanent global force produces the same acceleration before and after body rotation.

    A global +X force is applied. After 100 steps the body is rotated 180deg about Z.
    The acceleration (delta_v per phase) should be the same in both phases because the
    force is in the global frame and should not rotate with the body.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")
        mass = float(wp.to_torch(cube_object.root_view.get_masses())[0])
        com = cube_object.data.body_com_pos_w.torch.clone()

        # Apply permanent global force along +X at CoM
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=com,
            body_ids=body_ids,
            is_global=True,
        )

        # Phase 1: run N_STEPS
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase1 = cube_object.data.root_lin_vel_w.torch[0].clone()

        # Rotate body 180deg about Z (quat wxyz = [0, 0, 0, 1]) while keeping velocity
        root_pose = cube_object.data.root_state_w.torch[0, :7].clone().unsqueeze(0)
        root_pose[0, 3:7] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)  # 180deg about Z (xyzw)
        cube_object.write_root_pose_to_sim(root_pose)

        # Phase 2: run N_STEPS more
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase2 = cube_object.data.root_lin_vel_w.torch[0].clone()

        # Acceleration should be same in both phases: delta_v_phase2 ≈ delta_v_phase1
        delta_v_phase1 = vel_after_phase1[0].item()  # vx after phase 1
        delta_v_phase2 = vel_after_phase2[0].item() - vel_after_phase1[0].item()  # vx gained in phase 2

        expected_dv = FORCE_MAGNITUDE / mass * sim.cfg.dt * N_STEPS

        torch.testing.assert_close(
            torch.tensor(delta_v_phase1),
            torch.tensor(expected_dv),
            rtol=0.001,
            atol=0.0001,
        )
        torch.testing.assert_close(
            torch.tensor(delta_v_phase2),
            torch.tensor(expected_dv),
            rtol=0.001,
            atol=0.0001,
        )

        # Y and Z velocity should remain ~0
        assert abs(vel_after_phase2[1].item()) < 0.5, f"Unexpected Y velocity: {vel_after_phase2[1].item()}"
        assert abs(vel_after_phase2[2].item()) < 0.5, f"Unexpected Z velocity: {vel_after_phase2[2].item()}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_local_force_follows_rotation(device):
    """Test that a permanent local force rotates with the body.

    A local +X force is applied. After 100 steps the body is rotated 180deg about Z.
    Since local +X is now world -X, the force should decelerate the body back towards zero velocity.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Apply permanent local force along body +X
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=False,
        )

        # Phase 1: run N_STEPS — object accelerates along world +X
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase1 = cube_object.data.root_lin_vel_w.torch[0].clone()
        assert vel_after_phase1[0].item() > 1.0, "Object should be moving in +X"

        # Rotate body 180deg about Z while keeping velocity
        root_pose = cube_object.data.root_state_w.torch[0, :7].clone().unsqueeze(0)
        root_pose[0, 3:7] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)  # 180deg about Z (xyzw)
        cube_object.write_root_pose_to_sim(root_pose)

        # Phase 2: run N_STEPS — local +X is now world -X, so force decelerates
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase2 = cube_object.data.root_lin_vel_w.torch[0].clone()

        # Velocity should be approximately zero: decelerated by the same amount as it accelerated
        torch.testing.assert_close(
            vel_after_phase2[0],
            torch.tensor(0.0, device=device),
            atol=0.0001,
            rtol=0.001,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_at_offset_generates_torque(device):
    """Test that a global force applied at an offset from CoM generates the expected torque.

    A global +X force applied at +1m Y offset from CoM should produce:
    - Linear acceleration in +X
    - Angular acceleration about -Z (from cross product: (0,1,0) × (10,0,0) = (0,0,-10))
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Force at offset: +1m in Y from CoM (global frame)
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE  # +X force

        torques = torch.zeros(1, len(body_ids), 3, device=device)

        # Position offset: CoM position + 1m in Y (global frame)
        com_pos = cube_object.data.body_com_pos_w.torch[:, body_ids, :3].clone()
        positions = com_pos.clone()
        positions[..., 1] += 1.0  # +1m Y offset

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            is_global=True,
        )

        # Run 50 steps
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        lin_vel = cube_object.data.root_lin_vel_w.torch[0]
        ang_vel = cube_object.data.root_ang_vel_w.torch[0]

        # Linear velocity in +X should be positive
        assert lin_vel[0].item() > 0.1, f"Expected positive X velocity, got {lin_vel[0].item()}"

        # Angular velocity about Z should be negative (cross product: r × F, r=(0,1,0), F=(10,0,0) -> (0,0,-10))
        assert ang_vel[2].item() < -0.1, f"Expected negative Z angular velocity, got {ang_vel[2].item()}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_torque_invariant_under_rotation(device):
    """Test that a permanent global torque produces the same angular acceleration before and after rotation.

    A global +Z torque is applied. After 100 steps the body is rotated 90deg about X.
    The angular acceleration (delta_omega per phase) about Z should be the same in both phases
    because the torque is in the global frame.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Apply permanent global torque about +Z
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        torques = torch.zeros(1, len(body_ids), 3, device=device)
        torques[..., 2] = TORQUE_MAGNITUDE

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=True,
        )

        # Phase 1: run N_STEPS
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        omega_z_after_phase1 = cube_object.data.root_ang_vel_w.torch[0, 2].clone().item()

        # Rotate body 90deg about X and zero out velocities so phase 2 starts from rest
        # (avoids gyroscopic cross-coupling at high omega)
        root_pose = cube_object.data.root_state_w.torch[0, :7].clone().unsqueeze(0)
        root_pose[0, 3:7] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=device)  # 90deg about X (xyzw)
        cube_object.write_root_pose_to_sim(root_pose)
        cube_object.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        # Phase 2: run N_STEPS from rest with different body orientation
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        omega_z_after_phase2 = cube_object.data.root_ang_vel_w.torch[0, 2].clone().item()

        # Both phases start from rest — angular acceleration about Z should be the same
        torch.testing.assert_close(
            torch.tensor(omega_z_after_phase1),
            torch.tensor(omega_z_after_phase2),
            rtol=0.001,
            atol=0.0001,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_torque_after_translation(device):
    """Test that global force torque updates dynamically when the body translates.

    Phase 1: Cube at (1,0,0). Global force F=(0,10,0) applied at explicit position (1,0,0).
      stored_torque = cross((1,0,0), (0,10,0)) = (0,0,10)
      correction = -cross((1,0,0), (0,10,0)) = (0,0,-10)
      net torque = 0 → no rotation, only linear acceleration in +Y.

    Phase 2: Teleport cube to origin (0,0,0), zero velocity, don't re-apply force.
      stored_torque = (0,0,10) (unchanged in buffer)
      correction = -cross((0,0,0), (0,10,0)) = (0,0,0)
      net torque = (0,0,10) → rotation about +Z.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, height=1.0, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Phase 1 setup: Move cube to (1, 0, 1) and apply force at (1, 0, 1)
        root_state = cube_object.data.root_state_w.torch.clone()
        root_state[0, 0] = 1.0  # x = 1
        root_state[0, 1] = 0.0  # y = 0
        root_state[0, 2] = 1.0  # z = 1
        root_state[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity quat (xyzw)
        root_state[0, 7:] = 0.0  # zero velocity
        cube_object.write_root_state_to_sim(root_state)

        # Step once to let the state settle
        sim.step()
        cube_object.update(sim.cfg.dt)

        # Get current CoM position for the force application point
        com_pos = cube_object.data.body_com_pos_w.torch[:, body_ids, :3].clone()

        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 1] = FORCE_MAGNITUDE  # +Y force
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=com_pos,
            body_ids=body_ids,
            is_global=True,
        )

        # Phase 1: run 50 steps — force at CoM, expect no rotation
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        ang_vel_phase1 = cube_object.data.root_ang_vel_w.torch[0].clone()
        lin_vel_phase1 = cube_object.data.root_lin_vel_w.torch[0].clone()

        # Should have linear velocity in +Y
        assert lin_vel_phase1[1].item() > 0.1, f"Expected positive Y velocity, got {lin_vel_phase1[1].item()}"

        # Angular velocity should be ~0 (force applied at CoM → no torque)
        assert abs(ang_vel_phase1[2].item()) < 0.1, (
            f"Expected ~0 Z angular velocity in phase 1, got {ang_vel_phase1[2].item()}"
        )

        # Phase 2: Teleport cube to origin, zero velocity, don't re-apply force
        root_state2 = cube_object.data.root_state_w.torch.clone()
        root_state2[0, 0] = 0.0  # x = 0
        root_state2[0, 1] = 0.0
        root_state2[0, 2] = 1.0  # z = 1
        root_state2[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state2[0, 7:] = 0.0  # zero velocity
        cube_object.write_root_state_to_sim(root_state2)

        # Step once to let state settle
        sim.step()
        cube_object.update(sim.cfg.dt)

        # Phase 2: run 50 steps — body at origin but stored torque = cross((1,0,1), (0,10,0)) = (-10,0,10)
        # correction = -cross((0,0,1), (0,10,0)) = -(0,0,0 - but wait, z=1)
        # Actually: stored = cross((com_x,com_y,com_z), (0,10,0))
        # After teleport: correction = -cross(new_pos, F), net torque ≠ 0 since positions differ
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        ang_vel_phase2 = cube_object.data.root_ang_vel_w.torch[0].clone()

        # The X component of position changed from ~1 to ~0, so torque about Z changes.
        # stored_torque_z = com_x * Fy = ~1 * 10 = ~10
        # After teleport, correction_z = -new_x * Fy = ~0 * 10 = ~0
        # net torque_z ≈ 10 → positive Z angular velocity
        assert ang_vel_phase2[2].item() > 0.5, (
            f"Expected positive Z angular velocity in phase 2, got {ang_vel_phase2[2].item()}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_torque_reverses_on_opposite_side(device):
    """Test that dynamic correction produces correct torque sign depending on body position.

    Phase 1: Cube at (-1, 0, 1). Global F=(0, 10, 0) at world point P=(0, 0, 1).
      net torque_z = cross(P - link_pos, F)_z = cross((1,0,0), (0,10,0))_z = +10
      → positive Z angular velocity

    Phase 2: Teleport cube to (+1, 0, 1), zero velocity, don't re-apply force.
      net torque_z = cross(P - link_pos, F)_z = cross((-1,0,0), (0,10,0))_z = -10
      → negative Z angular velocity
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, height=1.0, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Move cube to (-1, 0, 1)
        root_state = cube_object.data.root_state_w.torch.clone()
        root_state[0, 0] = -1.0
        root_state[0, 1] = 0.0
        root_state[0, 2] = 1.0
        root_state[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state[0, 7:] = 0.0
        cube_object.write_root_state_to_sim(root_state)
        sim.step()
        cube_object.update(sim.cfg.dt)

        # Apply permanent global F=(0, 10, 0) at world point P=(0, 0, 1)
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 1] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)
        positions = torch.zeros(1, len(body_ids), 3, device=device)
        positions[..., 2] = 1.0  # P = (0, 0, 1)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            is_global=True,
        )

        # Phase 1: run 50 steps — expect positive Z angular velocity
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        omega_z_phase1 = cube_object.data.root_ang_vel_w.torch[0, 2].item()
        assert omega_z_phase1 > 0.1, f"Phase 1: expected positive omega_z, got {omega_z_phase1}"

        # Phase 2: Teleport cube to (+1, 0, 1), zero velocity
        root_state2 = cube_object.data.root_state_w.torch.clone()
        root_state2[0, 0] = 1.0
        root_state2[0, 1] = 0.0
        root_state2[0, 2] = 1.0
        root_state2[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state2[0, 7:] = 0.0
        cube_object.write_root_state_to_sim(root_state2)
        sim.step()
        cube_object.update(sim.cfg.dt)

        # Phase 2: run 50 steps — expect negative Z angular velocity
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        omega_z_phase2 = cube_object.data.root_ang_vel_w.torch[0, 2].item()
        assert omega_z_phase2 < -0.1, f"Phase 2: expected negative omega_z, got {omega_z_phase2}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_no_position_no_torque(device):
    """Test that global force without positions produces no torque (applied at CoM).

    A body at (2, 0, 1) with global F=(0, 10, 0) and no positions should experience
    only linear acceleration, no rotation. The force is applied at the body's CoM.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, height=1.0, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Move cube to (2, 0, 1)
        root_state = cube_object.data.root_state_w.torch.clone()
        root_state[0, 0] = 2.0
        root_state[0, 1] = 0.0
        root_state[0, 2] = 1.0
        root_state[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state[0, 7:] = 0.0
        cube_object.write_root_state_to_sim(root_state)
        sim.step()
        cube_object.update(sim.cfg.dt)

        # Apply global F=(0, 10, 0) WITHOUT positions → force at CoM, no torque
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 1] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=True,
        )

        # Run 50 steps
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        omega_z = cube_object.data.root_ang_vel_w.torch[0, 2].item()
        # No positions → force at CoM → zero torque → zero angular velocity
        assert abs(omega_z) < 0.01, f"Expected ~zero omega_z for force at CoM, got {omega_z}"

        # Should still have linear acceleration in +Y
        lin_vel_y = cube_object.data.root_lin_vel_w.torch[0, 1].item()
        assert lin_vel_y > 0.1, f"Expected positive Y velocity from applied force, got {lin_vel_y}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_multi_cube_different_torques_from_same_force(device):
    """Test kernel indexing across multiple envs with different CoM positions.

    2 cubes: Cube 0 at (-1, 0, 1), Cube 1 at (+1, 0, 1).
    Same global F=(0, 10, 0) at same world point P=(0, 0, 1) to both cubes.
    Cube 0: torque_z = cross((1,0,0), (0,10,0))_z = +10 → omega_z > 0
    Cube 1: torque_z = cross((-1,0,0), (0,10,0))_z = -10 → omega_z < 0
    Both have same linear acceleration in +Y.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=2, height=1.0, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Position cubes: Cube 0 at (-1, 0, 1), Cube 1 at (+1, 0, 1)
        root_state = cube_object.data.root_state_w.torch.clone()
        root_state[0, 0] = -1.0
        root_state[0, 1] = 0.0
        root_state[0, 2] = 1.0
        root_state[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state[0, 7:] = 0.0

        root_state[1, 0] = 1.0
        root_state[1, 1] = 0.0
        root_state[1, 2] = 1.0
        root_state[1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state[1, 7:] = 0.0
        cube_object.write_root_state_to_sim(root_state)
        sim.step()
        cube_object.update(sim.cfg.dt)

        # Apply same global F=(0, 10, 0) at P=(0, 0, 1) to both cubes
        forces = torch.zeros(2, len(body_ids), 3, device=device)
        forces[..., 1] = FORCE_MAGNITUDE
        torques = torch.zeros(2, len(body_ids), 3, device=device)
        positions = torch.zeros(2, len(body_ids), 3, device=device)
        positions[..., 2] = 1.0  # P = (0, 0, 1)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            is_global=True,
        )

        # Run 50 steps
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        # Cube 0: omega_z > 0 (force point is to the right of CoM)
        omega_z_0 = cube_object.data.root_ang_vel_w.torch[0, 2].item()
        assert omega_z_0 > 0.1, f"Cube 0: expected positive omega_z, got {omega_z_0}"

        # Cube 1: omega_z < 0 (force point is to the left of CoM)
        omega_z_1 = cube_object.data.root_ang_vel_w.torch[1, 2].item()
        assert omega_z_1 < -0.1, f"Cube 1: expected negative omega_z, got {omega_z_1}"

        # Both cubes should have same linear velocity in +Y (same force magnitude)
        lin_vel_y_0 = cube_object.data.root_lin_vel_w.torch[0, 1].item()
        lin_vel_y_1 = cube_object.data.root_lin_vel_w.torch[1, 1].item()
        assert abs(lin_vel_y_0 - lin_vel_y_1) < 0.5, (
            f"Both cubes should have similar Y velocity, got {lin_vel_y_0} and {lin_vel_y_1}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_torque_far_from_origin(device):
    """Test that global force torque correction produces correct physics at large world coordinates.

    Two cubes with identical relative geometry (force offset = (1, 0, 0) from CoM):
      Cube 0 at (0, 0, 1)    — near origin (reference)
      Cube 1 at (2000, 0, 1) — far from origin

    Both get global F=(0, 10, 0) at offset (1, 0, 0) from their respective CoMs.
    Expected torque: cross((1,0,0), (0,10,0)) = (0, 0, 10) for both.

    The compose kernel computes cross(P, F) - cross(link_pos, F):
      Cube 0: cross((1,0,1), F) - cross((0,0,1), F) — small values, no cancellation
      Cube 1: cross((2001,0,1), F) - cross((2000,0,1), F) — large values nearly cancel

    Both cubes should produce the same angular and linear velocities.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=2, height=1.0, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Position cubes: Cube 0 near origin, Cube 1 far from origin
        root_state = cube_object.data.root_state_w.torch.clone()
        # Cube 0 at (0, 0, 1)
        root_state[0, 0] = 0.0
        root_state[0, 1] = 0.0
        root_state[0, 2] = 1.0
        root_state[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state[0, 7:] = 0.0
        # Cube 1 at (2000, 0, 1)
        root_state[1, 0] = 2000.0
        root_state[1, 1] = 0.0
        root_state[1, 2] = 1.0
        root_state[1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity (xyzw)
        root_state[1, 7:] = 0.0
        cube_object.write_root_state_to_sim(root_state)
        sim.step()
        cube_object.update(sim.cfg.dt)

        # Apply F=(0, 10, 0) at +1m X offset from each cube's CoM
        forces = torch.zeros(2, len(body_ids), 3, device=device)
        forces[..., 1] = FORCE_MAGNITUDE  # +Y force
        torques = torch.zeros(2, len(body_ids), 3, device=device)

        # Positions: each cube's CoM + (1, 0, 0)
        com_pos = cube_object.data.body_com_pos_w.torch[:, body_ids, :3].clone()
        positions = com_pos.clone()
        positions[..., 0] += 1.0  # +1m X offset from CoM

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=body_ids,
            is_global=True,
        )

        # Run 50 steps
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        # Both cubes should have positive omega_z (cross((1,0,0), (0,10,0)) = (0,0,10))
        omega_z_0 = cube_object.data.root_ang_vel_w.torch[0, 2].item()
        omega_z_1 = cube_object.data.root_ang_vel_w.torch[1, 2].item()
        assert omega_z_0 > 0.1, f"Cube 0: expected positive omega_z, got {omega_z_0}"
        assert omega_z_1 > 0.1, f"Cube 1: expected positive omega_z, got {omega_z_1}"

        # omega_z values should match within 1% (same relative geometry)
        torch.testing.assert_close(
            torch.tensor(omega_z_0),
            torch.tensor(omega_z_1),
            rtol=0.01,
            atol=0.0,
            msg=lambda msg: (
                f"Angular velocity mismatch between near-origin and far-from-origin cubes:\n"
                f"  Cube 0 (near): omega_z = {omega_z_0:.6f}\n"
                f"  Cube 1 (far):  omega_z = {omega_z_1:.6f}\n{msg}"
            ),
        )

        # Linear velocity in +Y should also match
        lin_vel_y_0 = cube_object.data.root_lin_vel_w.torch[0, 1].item()
        lin_vel_y_1 = cube_object.data.root_lin_vel_w.torch[1, 1].item()
        torch.testing.assert_close(
            torch.tensor(lin_vel_y_0),
            torch.tensor(lin_vel_y_1),
            rtol=0.01,
            atol=0.0,
            msg=lambda msg: (
                f"Linear velocity mismatch between near-origin and far-from-origin cubes:\n"
                f"  Cube 0 (near): lin_vel_y = {lin_vel_y_0:.6f}\n"
                f"  Cube 1 (far):  lin_vel_y = {lin_vel_y_1:.6f}\n{msg}"
            ),
        )


@pytest.mark.parametrize("device", ["cuda:0"])
def test_global_force_no_position_no_rotation_large_offset(device):
    """Test that a global force without positions produces no rotation at large offsets.

    A cube is placed at (2000, 0, 1) and a global force F=(0, 10, 0) is applied
    without positions. The cube should accelerate linearly but not rotate.
    Before the fix, this would produce torque proportional to 2000 and cause rotation.
    """
    with build_simulation_context(
        device=device, add_ground_plane=False, auto_add_lighting=True, gravity_enabled=False
    ) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, height=1.0, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Place cube at large X offset
        root_state = cube_object.data.default_root_state.torch.clone()
        root_state[0, 0] = 2000.0  # large X position
        root_state[0, 1] = 0.0
        root_state[0, 2] = 1.0
        cube_object.write_root_pose_to_sim(root_state[:, :7])
        cube_object.write_root_velocity_to_sim(root_state[:, 7:])
        cube_object.reset()

        # Apply global force without positions (should go to CoM, no torque)
        forces = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=device)
        forces[0, :, 1] = 10.0  # F_y = 10 N

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            body_ids=body_ids,
            is_global=True,
        )

        # Step simulation
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        # Check: angular velocity should be near zero (no rotation)
        ang_vel = cube_object.data.root_ang_vel_w.torch[0]
        assert torch.allclose(ang_vel, torch.zeros(3, device=device), atol=0.01), (
            f"Expected near-zero angular velocity, got {ang_vel}. "
            "Global force without positions should not produce torque."
        )

        # Check: linear velocity in Y should be positive (force is in +Y)
        lin_vel = cube_object.data.root_lin_vel_w.torch[0]
        assert lin_vel[1] > 0.1, f"Expected positive Y velocity from applied force, got {lin_vel[1]}"


@pytest.mark.parametrize("device", ["cuda:0"])
def test_global_force_at_com_position_no_rotation_large_offset(device):
    """Test that a global force with position at CoM produces no rotation at large offsets.

    A cube is placed at (2000, 0, 1) and a global force F=(0, 10, 0) is applied
    at the cube's position (i.e., at its CoM). This should produce zero torque,
    serving as a control test alongside test_global_force_no_position_no_rotation_large_offset.
    """
    with build_simulation_context(
        device=device, add_ground_plane=False, auto_add_lighting=True, gravity_enabled=False
    ) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, height=1.0, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Place cube at large X offset
        root_state = cube_object.data.default_root_state.torch.clone()
        root_state[0, 0] = 2000.0
        root_state[0, 1] = 0.0
        root_state[0, 2] = 1.0
        cube_object.write_root_pose_to_sim(root_state[:, :7])
        cube_object.write_root_velocity_to_sim(root_state[:, 7:])
        cube_object.reset()

        # Apply global force AT the cube's position (torque should cancel)
        forces = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=device)
        forces[0, :, 1] = 10.0

        positions = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=device)
        positions[0, :, 0] = 2000.0
        positions[0, :, 2] = 1.0

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            positions=positions,
            body_ids=body_ids,
            is_global=True,
        )

        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        # Force at CoM → no rotation
        ang_vel = cube_object.data.root_ang_vel_w.torch[0]
        assert torch.allclose(ang_vel, torch.zeros(3, device=device), atol=0.01), (
            f"Expected near-zero angular velocity, got {ang_vel}. "
            "Global force at CoM position should not produce torque."
        )

        lin_vel = cube_object.data.root_lin_vel_w.torch[0]
        assert lin_vel[1] > 0.1, f"Expected positive Y velocity from applied force, got {lin_vel[1]}"
