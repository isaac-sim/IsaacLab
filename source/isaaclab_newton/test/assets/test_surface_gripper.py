# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public suction commands, physical acquisition, and reset/rebinding contracts."""

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg

from isaaclab.assets import SurfaceGripper, SurfaceGripperCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_box_acquisition_lift_release_and_rebind(device):
    """Payloads lift after hard resets, release on command, and reset independently."""
    legacy_cfg = SurfaceGripperCfg(*([None] * 8), 0.01, 5.0, 3.0, 0.1)
    assert (
        legacy_cfg.max_grip_distance,
        legacy_cfg.coaxial_force_limit,
        legacy_cfg.shear_force_limit,
        legacy_cfg.retry_interval,
    ) == (0.01, 5.0, 3.0, 0.1)
    cfg = SimulationCfg(
        device=device,
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, integrator="implicitfast", cone="elliptic"),
            use_cuda_graph=True,
        ),
    )
    half_sizes = np.asarray(((0.02, 0.02, 0.02), (0.075, 0.045, 0.0425)))
    with build_simulation_context(sim_cfg=cfg) as sim:
        manager = sim.backend_manager
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics))
        for world, (hx, hy, hz) in enumerate(half_sizes):
            builder.begin_world()
            root = f"/World/envs/env_{world}"
            builder.add_shape_box(-1, hx=2, hy=2, hz=0.05, xform=wp.transform((0, 0, -0.05), wp.quat_identity()))
            tool = builder.add_body(
                xform=wp.transform((0, 0, 2 * hz + 0.155), wp.quat(1, 0, 0, 0)),
                is_kinematic=True,
                label=f"{root}/Robot/panda_hand",
            )
            builder.add_shape_sphere(tool, radius=0.01)
            for index, x in enumerate((0.5, 0.0)):
                parcel = builder.add_body(
                    xform=wp.transform((x, 0, hz), wp.quat_identity()), label=f"{root}/Cube{index}"
                )
                builder.add_shape_box(
                    parcel,
                    hx=hx,
                    hy=hy,
                    hz=hz,
                    cfg=builder.ShapeConfig(density=float((0.03, 0.2)[world] / (8 * hx * hy * hz))),
                )
            builder.end_world()
        gripper = SurfaceGripper(
            SurfaceGripperCfg(prim_path=r".*/Robot/panda_hand", target_prim_path=r".*/Cube[0-9]+"), sim=sim
        )
        scene = InteractiveScene(InteractiveSceneCfg(num_envs=2, env_spacing=0.0))
        scene.surface_grippers["suction"] = gripper

        def step_tools(lift, vertical_velocity=0.0):
            # Drive standalone kinematic tools; tasks drive the robot through actuators.
            state = manager.backend.state_0
            pose, velocity = state.body_q.numpy(), state.body_qd.numpy()
            pose[tools, 2] = 2 * half_sizes[:, 2] + 0.105 + lift
            velocity[tools, 2] = vertical_velocity
            state.body_q.assign(pose)
            state.body_qd.assign(velocity)
            sim.step(render=False)

        try:
            for reset_index in range(2):
                sim.reset()
                model = manager.backend.model
                tools = [body for body, label in enumerate(model.body_label) if label.endswith("/panda_hand")]
                parcels = [body for body, label in enumerate(model.body_label) if label.endswith("/Cube1")]
                distant = [body for body, label in enumerate(model.body_label) if label.endswith("/Cube0")]
                if reset_index:
                    gripper.set_target_ids(torch.ones(2, dtype=torch.long, device=device))
                gripper.set_grippers_command_mask(
                    wp.ones(2, dtype=wp.float32, device=device),
                    wp.full(2, True, dtype=wp.bool, device=device),
                )
                gripper.write_data_to_sim()
                for step in range(400):
                    step_tools(min(max(step - 100, 0), 250) * 0.0006, 0.072 if 100 <= step < 350 else 0.0)
                assert wp.to_torch(gripper.state).tolist() == [1.0, 1.0]
                lifted = manager.backend.state_0.body_q.numpy()[parcels, 2].copy()
                assert np.all(lifted > half_sizes[:, 2] + 0.12)
                assert np.all(manager.backend.state_0.body_q.numpy()[distant, 2] < half_sizes[:, 2] + 0.01)
                if not reset_index:
                    scene.reset_to(
                        {"gripper": {"suction": -torch.ones(1, device=device)}},
                        env_ids=torch.tensor([0], device=device),
                    )
                    for _ in range(60):
                        step_tools(0.15)
                    assert wp.to_torch(gripper.state).tolist() == [-1.0, 1.0]
                    assert manager.backend.state_0.body_q.numpy()[parcels[1], 2] > half_sizes[1, 2] + 0.12
            gripper.reset(env_mask=wp.array([True, False], dtype=wp.bool, device=device))
            for _ in range(30):
                step_tools(0.15)
            assert wp.to_torch(gripper.state).tolist() == [-1.0, 1.0]
            gripper.set_grippers_command(-torch.ones(2, device=device))
            gripper.write_data_to_sim()
            for _ in range(200):
                step_tools(0.15)
            assert (wp.to_torch(gripper.state) == -1.0).all()
            assert np.all(manager.backend.state_0.body_q.numpy()[parcels, 2] < lifted - 0.08)
        finally:
            scene.close()
