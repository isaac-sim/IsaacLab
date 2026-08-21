# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real PhysX parity coverage for :class:`WrenchComposer`."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, device="cpu").app

import math

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.integration

_ROTATION_45_Z = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))


def _make_dual_cube_scene(device: str) -> tuple[RigidObject, RigidObject]:
    """Create matched composer and raw-PhysX cubes with a rotated initial pose."""
    for name, y_offset in (("Composer", 0.0), ("Raw", 3.0)):
        sim_utils.create_prim(f"/World/{name}", "Xform", translation=(0.0, y_offset, 1.0))

    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )
    composer = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/Composer/Object",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=_ROTATION_45_Z),
        )
    )
    raw = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/Raw/Object",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3.0, 1.0), rot=_ROTATION_45_Z),
        )
    )
    return composer, raw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PhysX wrench delivery requires CUDA-pinned staging")
def test_rotated_global_force_at_position_matches_physx_delivery() -> None:
    """Deliver a rotated global force at an offset and match raw PhysX velocity changes."""
    device = "cuda:0"
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        composer, raw = _make_dual_cube_scene(device)
        sim.reset()

        body_ids, _ = composer.find_bodies(".*")
        force = torch.tensor([[[0.0, 0.0, 10.0]]], device=device)
        torque = torch.zeros_like(force)
        world_offset = torch.tensor([[[0.0, 1.0, 0.0]]], device=device)
        composer_position = composer.data.body_com_pos_w.torch[:, body_ids, :3] + world_offset
        raw_position = raw.data.body_com_pos_w.torch[:, body_ids, :3] + world_offset

        composer.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force,
            torques=torque,
            positions=composer_position,
            body_ids=body_ids,
            is_global=True,
        )

        for _ in range(4):
            composer.write_data_to_sim()
            raw.root_view.apply_forces_and_torques_at_position(
                force_data=wp.from_torch(force.view(-1, 3).contiguous(), dtype=wp.float32),
                torque_data=wp.from_torch(torque.view(-1, 3).contiguous(), dtype=wp.float32),
                position_data=wp.from_torch(raw_position.view(-1, 3).contiguous(), dtype=wp.float32),
                indices=raw._ALL_INDICES,
                is_global=True,
            )
            sim.step()
            composer.update(sim.cfg.dt)
            raw.update(sim.cfg.dt)

        torch.testing.assert_close(
            composer.data.root_lin_vel_w.torch, raw.data.root_lin_vel_w.torch, rtol=1.0e-4, atol=1.0e-4
        )
        torch.testing.assert_close(
            composer.data.root_ang_vel_w.torch, raw.data.root_ang_vel_w.torch, rtol=1.0e-4, atol=1.0e-4
        )
        assert torch.abs(composer.data.root_ang_vel_w.torch[0, :2]).max().item() > 0.01
