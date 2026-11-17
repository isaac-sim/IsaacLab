# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public conveyor commands, native transport, and reset/teardown contracts."""

from functools import partial

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg

from isaaclab.controllers import Conveyor, ConveyorCfg, ConveyorSurfaceCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

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
        driver = Conveyor(
            ConveyorCfg(
                surface_specs=(
                    ConveyorSurfaceCfg("{ENV_REGEX_NS}/BeltA", velocity=0.2),
                    ConveyorSurfaceCfg(
                        "{ENV_REGEX_NS}/BeltB",
                        velocity=0.2,
                        curved=True,
                        direction=(0, 0, 1),
                        pivot_point=(-0.5, 0, 0),
                        radius=0.5,
                    ),
                ),
                body_pattern="/Cube$",
                body_count_per_env=1,
            ),
            2,
            sim=sim,
        )

        def transport():
            before = manager.backend.state_0.body_q.numpy().copy()
            for _ in range(120):
                sim.step(render=False)
            return manager.backend.state_0.body_q.numpy() - before

        try:
            sim.reset()
            motion = transport()
            assert motion[0, 0] > 0.025  # Straight belt in world 0.
            assert motion[2, 1] > 0.025  # Pivot belt in world 1.
            np.testing.assert_allclose(motion[[1, 3], :2], 0.0, atol=0.002)

            driver.set_speed_scale(-1.0)

            def reset_motion(reset):
                sim.step(render=False)  # Reversal produces a pending traction force in both worlds.
                before = manager.backend.state_0.body_qd.numpy().copy()
                reset()
                sim.step(render=False)
                return manager.backend.state_0.body_qd.numpy() - before

            selected_ids = wp.array([0], dtype=wp.int64, device=device)
            delta = reset_motion(lambda: driver.reset(env_ids=selected_ids))
            np.testing.assert_allclose(delta[0, :2], 0.0, atol=0.001)
            assert np.linalg.norm(delta[2, :2]) > 0.001
            mask = wp.array([False, True], dtype=wp.bool, device=device)
            driver.reset(env_mask=mask)  # Warm the full native mask path before capture.
            reset = partial(driver.reset, env_mask=mask)
            if device != "cpu":
                with wp.ScopedCapture(device=device) as capture:
                    driver.reset(env_mask=mask)
                reset = partial(wp.capture_launch, capture.graph)
            delta = reset_motion(reset)
            np.testing.assert_allclose(delta[2, :2], 0.0, atol=0.001)
            assert np.linalg.norm(delta[0, :2]) > 0.001
            mask.assign(np.array([True, False]))
            delta = reset_motion(reset)
            np.testing.assert_allclose(delta[0, :2], 0.0, atol=0.001)
            assert np.linalg.norm(delta[2, :2]) > 0.001

            driver.set_speed_scale(-1.0, env_ids=(0,))
            driver.stop(env_ids=[1])
            sim.reset()  # Rebind the model while preserving both selected-world commands.
            assert driver.speed_scale.tolist() == [-1.0, 0.0]
            motion = transport()
            assert motion[0, 0] < -0.025
            np.testing.assert_allclose(motion[1:, :2], 0.0, atol=0.002)

            driver.start(env_ids=[1])
            assert driver.speed_scale.tolist() == [-1.0, 1.0]
            assert transport()[2, 1] > 0.025
            driver.close()
            sim.reset()
            np.testing.assert_allclose(transport()[:, :2], 0.0, atol=0.002)
        finally:
            driver.close()
