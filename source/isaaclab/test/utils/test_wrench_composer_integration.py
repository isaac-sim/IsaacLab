# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the permanent wrench composer on rigid objects.

These tests validate that world-frame wrenches stay fixed in the world while body-frame wrenches follow the body,
and that the torque of a world-frame force applied at a world position tracks the body's pose.
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

pytestmark = pytest.mark.integration

N_STEPS = 50
FORCE_MAGNITUDE = 10.0
TORQUE_MAGNITUDE = 1.0
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def spawn_cubes(num_cubes: int, device: str) -> RigidObject:
    """Spawn ``num_cubes`` free-floating cubes, one per environment, at 1 m height."""
    for i in range(num_cubes):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=(float(i), 0.0, 1.0))
    cfg = RigidObjectCfg(
        prim_path="/World/Table_[^/]*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    return RigidObject(cfg=cfg)


def step(sim, cube: RigidObject, num_steps: int = N_STEPS) -> None:
    for _ in range(num_steps):
        cube.write_data_to_sim()
        sim.step()
        cube.update(sim.cfg.dt)


def teleport(sim, cube: RigidObject, env_id: int, pos=None, quat=IDENTITY_QUAT, zero_velocity: bool = True) -> None:
    """Write a new root pose (and optionally zero velocity) for one environment and let the state settle."""
    root_pose = cube.data.root_pose_w.torch.clone()
    if pos is not None:
        root_pose[env_id, :3] = torch.tensor(pos, device=cube.device)
    root_pose[env_id, 3:7] = torch.tensor(quat, device=cube.device)
    cube.write_root_pose_to_sim_index(root_pose=root_pose)
    if zero_velocity:
        root_vel = cube.data.root_vel_w.torch.clone()
        root_vel[env_id] = 0.0
        cube.write_root_velocity_to_sim_index(root_velocity=root_vel)
    sim.step()
    cube.update(sim.cfg.dt)


def set_permanent_wrench(cube: RigidObject, *, forces=None, torques=None, positions=None, is_global: bool) -> None:
    num_bodies = cube.num_bodies
    zeros = torch.zeros(cube.num_instances, num_bodies, 3, device=cube.device)
    cube.permanent_wrench_composer.set_forces_and_torques_index(
        forces=zeros if forces is None else forces,
        torques=zeros if torques is None else torques,
        positions=positions,
        body_ids=list(range(num_bodies)),
        is_global=is_global,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("kind", ["force", "torque"])
def test_global_wrench_is_invariant_under_body_rotation(device, kind):
    """A world-frame force (torque) accelerates the body identically before and after the body is rotated."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube = spawn_cubes(1, device)
        sim.reset()

        wrench = torch.zeros(1, cube.num_bodies, 3, device=device)
        if kind == "force":
            wrench[..., 0] = FORCE_MAGNITUDE
            set_permanent_wrench(cube, forces=wrench, positions=cube.data.body_com_pos_w.torch.clone(), is_global=True)
            velocity = lambda: cube.data.root_lin_vel_w.torch[0]  # noqa: E731
            axis = 0
            rotated_quat = (0.0, 0.0, 1.0, 0.0)  # 180 degrees about Z
        else:
            wrench[..., 2] = TORQUE_MAGNITUDE
            set_permanent_wrench(cube, torques=wrench, is_global=True)
            velocity = lambda: cube.data.root_ang_vel_w.torch[0]  # noqa: E731
            axis = 2
            rotated_quat = (0.7071, 0.0, 0.0, 0.7071)  # 90 degrees about X

        step(sim, cube)
        gained_phase1 = velocity()[axis].item()
        # restart from rest in the rotated orientation
        teleport(sim, cube, 0, quat=rotated_quat)
        step(sim, cube)
        gained_phase2 = velocity()[axis].item()

        assert gained_phase1 > 0.1
        torch.testing.assert_close(gained_phase2, gained_phase1, rtol=1e-3, atol=1e-4)
        if kind == "force":
            mass = float(wp.to_torch(cube.root_view.get_masses())[0])
            expected = FORCE_MAGNITUDE / mass * sim.cfg.dt * N_STEPS
            torch.testing.assert_close(gained_phase1, expected, rtol=1e-3, atol=1e-4)
            assert velocity()[1:].abs().max().item() < 0.5


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_local_force_follows_body_rotation(device):
    """A body-frame +X force decelerates the body after a 180 degree turn, bringing it back to rest."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube = spawn_cubes(1, device)
        sim.reset()

        forces = torch.zeros(1, cube.num_bodies, 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        set_permanent_wrench(cube, forces=forces, is_global=False)

        step(sim, cube)
        assert cube.data.root_lin_vel_w.torch[0, 0].item() > 1.0, "Object should be moving in +X"
        teleport(sim, cube, 0, quat=(0.0, 0.0, 1.0, 0.0), zero_velocity=False)
        step(sim, cube)

        torch.testing.assert_close(cube.data.root_lin_vel_w.torch[0, 0].item(), 0.0, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_torque_tracks_body_pose(device):
    """The torque of a world-frame force at a world point is ``cross(point - com, force)`` for every body pose.

    Four cubes receive the same +Y force:

    - cube 0 at (-1, 0, 1) with the force at (0, 0, 1): lever arm +X, so it spins about +Z
    - cube 1 at (+1, 0, 1) with the force at (0, 0, 1): lever arm -X, so it spins about -Z
    - cube 2 at (2000, 0, 1) with no position: the force acts at the CoM, so it must not spin
    - cube 3 at (2000, 5, 1) with the force 1 m in +X of its CoM: same lever arm as cube 0, so the
      near-origin and far-from-origin cubes must move identically (no cancellation error)

    Teleporting cube 0 to (+1, 0, 1) afterwards flips its torque without re-applying the force.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube = spawn_cubes(4, device)
        sim.reset()
        for env_id, pos in enumerate([(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2000.0, 0.0, 1.0), (2000.0, 5.0, 1.0)]):
            teleport(sim, cube, env_id, pos=pos)

        forces = torch.zeros(4, cube.num_bodies, 3, device=device)
        forces[..., 1] = FORCE_MAGNITUDE
        positions = torch.zeros(3, cube.num_bodies, 3, device=device)
        positions[:2, :, 2] = 1.0
        positions[2] = cube.data.body_com_pos_w.torch[3, :, :3] + torch.tensor([1.0, 0.0, 0.0], device=device)
        composer = cube.permanent_wrench_composer
        body_ids = list(range(cube.num_bodies))
        composer.set_forces_and_torques_index(
            forces=forces[[0, 1, 3]], positions=positions, body_ids=body_ids, env_ids=[0, 1, 3], is_global=True
        )
        composer.add_forces_and_torques_index(forces=forces[[2]], body_ids=body_ids, env_ids=[2], is_global=True)

        step(sim, cube)
        omega_z = cube.data.root_ang_vel_w.torch[:, 2]
        lin_vel_y = cube.data.root_lin_vel_w.torch[:, 1]
        assert omega_z[0].item() > 0.1 and omega_z[1].item() < -0.1
        assert cube.data.root_ang_vel_w.torch[2].abs().max().item() < 0.01
        assert torch.all(lin_vel_y > 0.1)
        torch.testing.assert_close(omega_z[3].item(), omega_z[0].item(), rtol=0.01, atol=0.0)
        torch.testing.assert_close(lin_vel_y[3].item(), lin_vel_y[0].item(), rtol=0.01, atol=0.0)

        teleport(sim, cube, 0, pos=(1.0, 0.0, 1.0))
        step(sim, cube)
        assert cube.data.root_ang_vel_w.torch[0, 2].item() < -0.1
