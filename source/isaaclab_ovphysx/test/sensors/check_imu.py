# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone validation script for the OVPhysX IMU sensor.

Run with:
    ./isaaclab.sh -p source/isaaclab_ovphysx/test/sensors/check_imu.py

Spawns a free-falling sphere with an IMU attached, steps the simulation for
100 steps, and prints the angular velocity and linear acceleration in the
IMU body frame. In freefall the proper acceleration is zero (only gravity
acts on the body), so the IMU reading should be approximately ``[0, 0, 0]``
— the finite-difference world-frame acceleration ``(-g)`` cancels the IMU's
gravity bias ``(+g)``.

Runs kitless under ``./isaaclab.sh -p`` — no AppLauncher needed.
"""

from __future__ import annotations

import torch
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.imu import Imu, ImuCfg
from isaaclab.sim import SimulationCfg, build_simulation_context


def main() -> None:
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device="cpu", dt=0.005)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        # Spawn one ball per env at /World/env_<i>/ball.
        spawn_cfg = sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
        # /World/env_<i> Xforms are siblings under /World — no envs container needed
        num_envs = 2
        for i in range(num_envs):
            sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i * 5.0, 0.0, 0.0))
            spawn_cfg.func(f"/World/env_{i}/ball", spawn_cfg, translation=(0.0, 0.0, 1.0))

        balls = RigidObject(
            RigidObjectCfg(
                prim_path="/World/env_*/ball",
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
            )
        )
        imu = Imu(ImuCfg(prim_path="/World/env_*/ball"))
        sim.reset()

        dt = sim.get_physics_dt()
        for step in range(100):
            balls.write_data_to_sim()
            sim.step()
            balls.update(dt)
            imu.update(dt, force_recompute=True)

            if step in (0, 10, 50, 99):
                ang = imu.data.ang_vel_b.torch
                lin = imu.data.lin_acc_b.torch
                print(f"step={step:3d}  ang_vel_b={ang[0].cpu().numpy()}  lin_acc_b={lin[0].cpu().numpy()}")

        # Final sanity check: in freefall the proper acceleration is zero
        # (gravity cancels the bias). The norm should be near 0.
        final_lin = imu.data.lin_acc_b.torch
        magnitude = torch.linalg.norm(final_lin, dim=1).cpu().numpy()
        print(f"final |lin_acc_b| per env = {magnitude}")


if __name__ == "__main__":
    main()
