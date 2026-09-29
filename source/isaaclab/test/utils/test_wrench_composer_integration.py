# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX delivery of :class:`~isaaclab.utils.wrench_composer.WrenchComposer` wrenches.

The composer arithmetic is covered by literal units in ``test_wrench_composer.py``. This module checks the two ways a
rigid object hands the composed wrench to PhysX: body-frame vectors for a world force at a world position, and
world-frame vectors applied at the center of mass.
"""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import math

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.integration

_ROTATION_45_Z = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
# A 120-degree turn about (1, 1, 1) maps the body axes x, y, z onto the world axes y, z, x.
_ROTATION_120_XYZ = (0.5, 0.5, 0.5, 0.5)
_NUM_STEPS = 4


def _make_cube(name: str, position: tuple[float, float, float], rotation: tuple[float, ...]) -> RigidObject:
    """Spawn a free, undamped unit-mass cube with a rotated initial pose."""
    spawn = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, linear_damping=0.0, angular_damping=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cfg = RigidObjectCfg(
        prim_path=f"/World/{name}", spawn=spawn, init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation)
    )
    return RigidObject(cfg=cfg)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_composed_wrenches_reach_physx_in_the_frame_they_are_delivered(device: str) -> None:
    """Match raw PhysX for a world force at an offset, and world-frame physics for a world wrench at the CoM."""
    with build_simulation_context(
        device=device, gravity_enabled=False, add_ground_plane=False, auto_add_lighting=True
    ) as sim:
        sim._app_control_on_stop_handle = None
        # Off the world z axis, so the CoM correction cross(com, F) of the composed torque is not zero.
        composed = _make_cube("Composed", (1.0, 0.0, 1.0), _ROTATION_45_Z)
        raw = _make_cube("Raw", (1.0, 3.0, 1.0), _ROTATION_45_Z)
        # Far from the origin, a world force at the CoM must not pick up a torque from the body position.
        world = _make_cube("World", (2000.0, 0.0, 1.0), _ROTATION_120_XYZ)
        sim.reset()

        # A world force at a world offset from the CoM is composed into body-frame force and induced torque.
        force = torch.tensor([[[0.0, 0.0, 10.0]]], device=device)
        offset = torch.tensor([[[0.0, 1.0, 0.0]]], device=device)
        composed_position = composed.data.body_com_pos_w.torch + offset
        raw_position = raw.data.body_com_pos_w.torch + offset
        composed.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force, positions=composed_position, is_global=True
        )
        assert composed.permanent_wrench_composer.get_forces_and_torques()[2] is False

        # A world force and torque at the CoM take the world-frame fast path without composition.
        world_force, world_torque = 10.0, 0.01
        world.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.tensor([[[world_force, 0.0, 0.0]]], device=device),
            torques=torch.tensor([[[0.0, 0.0, world_torque]]], device=device),
            is_global=True,
        )
        assert world.permanent_wrench_composer.get_forces_and_torques()[2] is True

        raw_force = wp.from_torch(force.view(-1, 3), dtype=wp.float32)
        raw_torque = wp.zeros((1, 3), dtype=wp.float32, device=device)
        raw_position_data = wp.from_torch(raw_position.view(-1, 3), dtype=wp.float32)
        for _ in range(_NUM_STEPS):
            composed.write_data_to_sim()
            world.write_data_to_sim()
            raw.root_view.apply_forces_and_torques_at_position(
                force_data=raw_force,
                torque_data=raw_torque,
                position_data=raw_position_data,
                indices=raw._ALL_INDICES,
                is_global=True,
            )
            sim.step()
            for asset in (composed, raw, world):
                asset.update(sim.cfg.dt)

        torch.testing.assert_close(
            composed.data.root_lin_vel_w.torch, raw.data.root_lin_vel_w.torch, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            composed.data.root_ang_vel_w.torch, raw.data.root_ang_vel_w.torch, rtol=1e-4, atol=1e-4
        )
        assert torch.abs(composed.data.root_ang_vel_w.torch[0, :2]).max().item() > 0.01

        # World-frame delivery: the velocities follow the world axes, not the rotated body axes.
        duration = _NUM_STEPS * sim.cfg.dt
        mass = world.data.body_mass.torch[0, 0].item()
        inertia_zz = world.data.body_inertia.torch[0, 0, 8].item()
        expected_lin_vel = torch.tensor([[world_force / mass * duration, 0.0, 0.0]], device=device)
        expected_ang_vel = torch.tensor([[0.0, 0.0, world_torque / inertia_zz * duration]], device=device)
        torch.testing.assert_close(world.data.root_lin_vel_w.torch, expected_lin_vel, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(world.data.root_ang_vel_w.torch, expected_ang_vel, rtol=1e-3, atol=1e-4)
