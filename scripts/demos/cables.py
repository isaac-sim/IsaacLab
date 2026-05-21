# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn cables welded to rigid plug bodies and let them settle.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/cables.py
    ./isaaclab.sh -p scripts/demos/cables.py --num_cables 10

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn cables welded to rigid plugs.")
parser.add_argument("--num_cables", type=int, default=4, help="Number of cable+plug pairs to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import random

import tqdm
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab_contrib.cable import CableAttachmentCfg, CableObject, CableObjectCfg


def y_axis_quat(angle_rad: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a rotation of ``angle_rad`` about +Y."""
    return (0.0, math.sin(0.5 * angle_rad), 0.0, math.cos(0.5 * angle_rad))


def design_scene(num_cables: int) -> dict[str, "CableObject | RigidObject"]:
    """Spawn ground, dome light, and N cable-plug pairs welded together."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    num_points = 20
    segment_length = 0.015
    cable_length = (num_points - 1) * segment_length
    width = 0.01
    xy_jitter = 0.3
    z_base = 0.8
    z_spacing = 1.5 * width

    print(f"[INFO]: Spawning {num_cables} cable+plug pairs...")
    entities: dict[str, CableObject | RigidObject] = {}
    for idx in tqdm.tqdm(range(num_cables)):
        angle = random.uniform(0.0, 2.0 * math.pi)
        cx = random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.cos(angle)
        cy = random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.sin(angle)
        cz = z_base + idx * z_spacing

        static_cfg = RigidObjectCfg(
            prim_path=f"/World/Origin/Static{idx:03d}",
            spawn=sim_utils.SphereCfg(
                radius=0.01,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx+num_points*segment_length, cy, cz+0.1),
                rot=y_axis_quat(math.pi / 2.0),
            ),
        )
        entities[f"Static{idx:03d}"] = RigidObject(cfg=static_cfg)

        plug_cfg = RigidObjectCfg(
            prim_path=f"/World/Origin/Plug{idx:03d}",
            spawn=sim_utils.CylinderCfg(
                radius=0.01,
                height=0.04,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx, cy, cz),
                rot=y_axis_quat(math.pi / 2.0),
            ),
        )
        entities[f"Plug{idx:03d}"] = RigidObject(cfg=plug_cfg)

        cable_cfg = CableObjectCfg(
            prim_path=f"/World/Origin/Cable{idx:03d}",
            spawn=sim_utils.CableCfg(
                positions=[(i * segment_length, 0.0, 0.0) for i in range(num_points)],
                width=width,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(random.random(), random.random(), random.random())
                ),
                physics_material=NewtonCableMaterialCfg(
                    stretch_stiffness=1e3,
                    bend_stiffness=1e-4,
                    stretch_damping=1e-1,
                    bend_damping=1e-4,
                    density=100.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=(cx, cy, cz)),
            attachments=[
                CableAttachmentCfg(
                    target_prim_path=f"/World/Origin/Plug{idx:03d}",
                    cable_anchor="tail",
                    cable_local_pos=(0.0, 0.0, 2*segment_length),
                ),
                CableAttachmentCfg(
                    target_prim_path=f"/World/Origin/Static{idx:03d}",
                    cable_anchor="head",
                    cable_local_pos=(0.0, 0.0, 0.0),
                ),
            ],
        )
        entities[f"Cable{idx:03d}"] = CableObject(cfg=cable_cfg)

    return entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, CableObject | RigidObject]):
    """Step the sim and periodically snap cables back to their initial state."""
    sim_dt = sim.get_physics_dt()
    reset_steps = int(2.0 / sim_dt)
    count = 0

    while simulation_app.is_running():
        if count % reset_steps == 0:
            count = 0
            sim.reset(soft=True)
            print("[INFO]: Resetting cable state...")
        sim.step()
        count += 1
        for entity in entities.values():
            entity.update(sim_dt)


def main():
    """Main entry point."""
    from isaaclab_newton.physics import NewtonCfg, NewtonCollisionPipelineCfg

    from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

    physics_cfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=20,
            rigid_body_contact_buffer_size=1024,
            rigid_contact_k_start=1.0e1,
            rigid_avbd_beta=1e2,
        ),
        num_substeps=8,
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
    )
    physics_cfg.model_cfg = NewtonModelCfg(
        shape_material_ke=1.0e3,
        shape_material_kd=1.0e0,
        shape_material_mu=1.0,
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=physics_cfg,
        visualizer_cfgs=[
            NewtonVisualizerCfg(eye=(0.5, 1.5, 0.5), lookat=(0.0, 0.0, 0.05)),
            KitVisualizerCfg(eye=(0.5, 1.5, 0.5), lookat=(0.0, 0.0, 0.05)),
        ],
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_entities = design_scene(num_cables=args_cli.num_cables)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities)
    print("[INFO]: Simulation complete...")


if __name__ == "__main__":
    main()
    simulation_app.close()
