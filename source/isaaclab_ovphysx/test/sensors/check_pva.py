# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone validation script for the OVPhysX PVA sensor.

Run with:
    ./isaaclab.sh -p source/isaaclab_ovphysx/test/sensors/check_pva.py

Spawns a free-falling sphere with a PVA sensor attached, steps the simulation
for 100 steps, and prints the world-frame position, body-frame linear/angular
velocities, body-frame linear/angular accelerations, and projected gravity.
In freefall the coordinate linear acceleration is ``-g`` rotated into the
body frame, so its magnitude should converge to ``~9.81 m/s^2``. The
projected gravity unit vector should remain ``≈ (0, 0, -1)`` for a body that
stays upright.

Runs kitless under ``./isaaclab.sh -p`` — no AppLauncher needed.
"""

from __future__ import annotations

import torch
from isaaclab_ovphysx.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.pva import Pva, PvaCfg
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
        pva = Pva(PvaCfg(prim_path="/World/env_*/ball"))
        sim.reset()

        dt = sim.get_physics_dt()
        for step in range(100):
            balls.write_data_to_sim()
            sim.step()
            balls.update(dt)
            pva.update(dt, force_recompute=True)

            if step in (0, 10, 50, 99):
                data = pva.data
                pos = data.pos_w.torch
                lin_v = data.lin_vel_b.torch
                ang_v = data.ang_vel_b.torch
                lin_a = data.lin_acc_b.torch
                ang_a = data.ang_acc_b.torch
                pg = data.projected_gravity_b.torch
                print(
                    f"step={step:3d}  pos_w={pos[0].cpu().numpy()}  lin_vel_b={lin_v[0].cpu().numpy()}  "
                    f"ang_vel_b={ang_v[0].cpu().numpy()}  lin_acc_b={lin_a[0].cpu().numpy()}  "
                    f"ang_acc_b={ang_a[0].cpu().numpy()}  proj_g={pg[0].cpu().numpy()}"
                )

        # Final sanity check: in freefall the coordinate acceleration magnitude is ~g.
        final_lin = pva.data.lin_acc_b.torch
        magnitude = torch.linalg.norm(final_lin, dim=1).cpu().numpy()
        print(f"final |lin_acc_b| per env = {magnitude}  (expected ~9.81)")
        # projected gravity unit vector magnitude is 1.
        final_pg = pva.data.projected_gravity_b.torch
        pg_magnitude = torch.linalg.norm(final_pg, dim=1).cpu().numpy()
        print(f"final |projected_gravity_b| per env = {pg_magnitude}  (expected ~1.0)")


if __name__ == "__main__":
    main()
