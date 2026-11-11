# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native transport, reset, and callback-lifetime contracts for the Newton adapter."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg

from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_contrib.conveyors.newton import SurfaceVelocity
from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_transport_reset_rebind_and_close(device):
    """Real straight/pivot traction survives hard resets and leaves no eager or captured callbacks."""
    cfg = SimulationCfg(
        device=device,
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, integrator="implicitfast", cone="elliptic"),
            use_cuda_graph=device != "cpu",
        ),
    )
    with build_simulation_context(sim_cfg=cfg) as sim:
        manager = sim.backend_manager
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics))
        for world in range(2):
            builder.begin_world()
            root = f"/World/envs/env_{world}"
            for name, y in (("B", 0.5), ("A", -0.5)):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((0, y, 0), wp.quat_identity()),
                    hx=1,
                    hy=0.35,
                    hz=0.1,
                    cfg=builder.ShapeConfig(mu=1.1e-5),
                    label=f"{root}/Belt{name}",
                )
            for name, x, y in (("Cube", 0, -0.5 if world == 0 else 0.5), ("Untracked", -0.5, -0.5)):
                body = builder.add_link(xform=wp.transform((x, y, 0.21), wp.quat_identity()), label=f"{root}/{name}")
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=builder.ShapeConfig(mu=1.1e-5))
                builder.add_articulation([builder.add_joint_free(body)])
            builder.end_world()
        driver = SurfaceVelocity(
            2,
            (
                SurfaceVelocitySpec("{ENV_REGEX_NS}/BeltA", velocity=0.2),
                SurfaceVelocitySpec(
                    "{ENV_REGEX_NS}/BeltB",
                    velocity=0.2,
                    curved=True,
                    direction=(0, 0, 1),
                    pivot_point=(-0.5, 0, 0),
                    radius=0.5,
                ),
            ),
            manager=manager,
            body_pattern="/Cube$",
            body_count_per_env=1,
        )
        try:
            for _ in range(2):
                sim.reset()
                before = manager.backend.state_0.body_q.numpy().copy()
                for _ in range(120):
                    sim.step(render=False)
                after = manager.backend.state_0.body_q.numpy()
                assert after[0, 0] - before[0, 0] > 0.025  # Straight belt in world 0.
                assert after[2, 1] - before[2, 1] > 0.025  # Pivot belt in world 1.
                np.testing.assert_allclose(after[[1, 3], :2], before[[1, 3], :2], atol=0.002)

            driver.close()
            sim.reset()
            before = manager.backend.state_0.body_q.numpy().copy()
            for _ in range(120):
                sim.step(render=False)
            np.testing.assert_allclose(manager.backend.state_0.body_q.numpy()[:, :2], before[:, :2], atol=0.002)
        finally:
            driver.close()
