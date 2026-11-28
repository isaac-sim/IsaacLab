# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend tests for OvPhysX scene gravity randomization."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.assets import RigidObject
from isaaclab_ov.physics import OvPhysxCfg
from ovphysx import PhysX

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp import randomize_physics_scene_gravity
from isaaclab.managers import EventTermCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.ov


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_gravity_event_changes_rigid_body_motion(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gravity events must change motion without resealing unchanged values or hiding failed updates."""
    sim_cfg = SimulationCfg(physics=PhysxAutoCfg(ovphysx=OvPhysxCfg()), device=device, dt=1.0 / 60.0)
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        cube = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Cube",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    mass_props=sim_utils.MassCfg(mass=1.0),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                ),
            )
        )
        sim.reset()
        dt = sim.get_physics_dt()
        nominal_gravity = sim.cfg.gravity
        cube.update(dt)

        env = SimpleNamespace(device=sim.device, sim=sim)
        event_cfg = EventTermCfg(
            func=randomize_physics_scene_gravity,
            params={
                "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "operation": "abs",
                "distribution": "gaussian",
            },
        )
        gravity_event = randomize_physics_scene_gravity(event_cfg, env)

        def step(count: int) -> None:
            initial_time = sim.backend_manager.get_simulation_time()
            for _ in range(count):
                cube.write_data_to_sim()
                sim.step()
                cube.update(dt)
            assert sim.backend_manager.get_simulation_time() == pytest.approx(initial_time + count * dt)

        def set_gravity(gravity: tuple[float, float, float]) -> None:
            gravity_event(
                env,
                env_ids=None,
                gravity_distribution_params=(gravity, (0.0, 0.0, 0.0)),
                operation="abs",
            )
            assert sim.backend_manager.get_gravity() == pytest.approx(gravity)
            assert sim.cfg.gravity == nominal_gravity

        set_gravity((0.0, 0.0, 0.0))
        initial_height = cube.data.root_pos_w.torch[0, 2].item()
        step(30)
        zero_gravity_height = cube.data.root_pos_w.torch[0, 2].item()

        set_gravity((0.0, 0.0, -9.81))
        # -9.81 is not exactly representable in float32; repeating it must still hit the cache.
        ordinal = sim.backend_manager.backend.scene.ordinal
        set_gravity((0.0, 0.0, -9.81))
        assert sim.backend_manager.backend.scene.ordinal == ordinal
        # Changing scene gravity does not wake sleeping PhysX actors. Reset terms
        # write body state after gravity randomization, so mirror that wake-up here.
        root_velocity = torch.zeros((1, 6), device=sim.device)
        root_velocity[0, 2] = 0.001
        cube.write_root_velocity_to_sim_index(root_velocity=root_velocity)
        step(30)
        earth_gravity_height = cube.data.root_pos_w.torch[0, 2].item()

        assert zero_gravity_height == pytest.approx(initial_height, abs=1.0e-4)
        assert earth_gravity_height - zero_gravity_height < -0.5
        assert cube.data.root_lin_vel_w.torch[0, 2].item() == pytest.approx(0.001 - 9.81 * 30 * dt, abs=0.02)

        # A failed update may already have changed native gravity. Re-requesting
        # the last successful value must restore it rather than hit the cache.
        wait_task = PhysX.wait_task

        def finish_then_fail(physx: PhysX, operation: object) -> None:
            wait_task(physx, operation)
            raise RuntimeError("control completion failed")

        with monkeypatch.context() as failed_update:
            failed_update.setattr(PhysX, "wait_task", finish_then_fail)
            with pytest.raises(RuntimeError, match="control completion failed"):
                set_gravity((0.0, 0.0, 0.0))
        set_gravity((0.0, 0.0, -9.81))
        cube.write_root_velocity_to_sim_index(root_velocity=root_velocity)
        step(30)
        assert cube.data.root_lin_vel_w.torch[0, 2].item() == pytest.approx(0.001 - 9.81 * 30 * dt, abs=0.02)
