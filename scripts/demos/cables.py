# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a pile of cables at varied z-axis rotations so they collide and settle on each other.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/cables.py
    ./isaaclab.sh -p scripts/demos/cables.py --num_cables 40

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn a pile of cables at varied z-axis rotations.")
parser.add_argument("--num_cables", type=int, default=25, help="Number of cables to spawn.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import random

import tqdm
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils

from isaaclab_contrib.cable import CableObject, CableObjectCfg


def z_axis_quat(angle_rad: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a rotation of ``angle_rad`` about +Z."""
    return (0.0, 0.0, math.sin(0.5 * angle_rad), math.cos(0.5 * angle_rad))


def design_scene(num_cables: int) -> dict[str, CableObject]:
    """Spawn a ground plane, a dome light, and a pile of randomly oriented cables."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    # Cable centerline: 10 control points along local +X, length ~0.9 m.
    num_points = 20
    segment_length = 0.015
    cable_length = (num_points - 1) * segment_length
    width = 0.01

    # Pile footprint: small XY box, stacked Z so cables fall and intersect.
    # Spacing is generous to avoid self-contact at spawn, and the base height is
    # kept low so cables don't gain a lot of velocity before first contact.
    xy_jitter = 0.3
    z_spacing = 1.5 * width
    z_base = 0.8

    print(f"[INFO]: Spawning {num_cables} cables...")
    entities: dict[str, CableObject] = {}
    for idx in tqdm.tqdm(range(num_cables)):
        angle = random.uniform(0.0, 2.0 * math.pi)
        cx = random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.cos(angle)
        cy = random.uniform(-xy_jitter, xy_jitter) - 0.5 * cable_length * math.sin(angle)
        cz = z_base + idx * z_spacing

        spawn_cfg = sim_utils.CableCfg(
            positions=[(i * segment_length, 0.0, 0.0) for i in range(num_points)],
            width=width,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(random.random(), random.random(), random.random())
            ),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e6,
                bend_stiffness=1e-4,
                stretch_damping=1e-4,
                bend_damping=1e-4,
                density=100.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        cfg = CableObjectCfg(
            prim_path=f"/World/Origin/Cable{idx:03d}",
            spawn=spawn_cfg,
            init_state=CableObjectCfg.InitialStateCfg(pos=(cx, cy, cz), rot=z_axis_quat(angle)),
        )
        entities[f"Cable{idx:03d}"] = CableObject(cfg=cfg)

        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path="/home/mmichelis/Documents/IsaacLab-Origin/scripts/demos/cable001.usda",
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e6,
                bend_stiffness=1e-4,
                stretch_damping=1e-4,
                bend_damping=1e-4,
                density=100.0,
            ),
        )
        cfg = CableObjectCfg(
            prim_path=f"/World/Origin/Cable1{idx:03d}",
            spawn=spawn_cfg,
            init_state=CableObjectCfg.InitialStateCfg(pos=(cx, cy, cz), rot=z_axis_quat(angle)),
        )
        entities[f"Cable1{idx:03d}"] = CableObject(cfg=cfg)

        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path="/home/mmichelis/Documents/IsaacLab-Origin/scripts/demos/cable002.usda",
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e6,
                bend_stiffness=1e-4,
                stretch_damping=1e-4,
                bend_damping=1e-4,
                density=100.0,
            ),
        )
        cfg = CableObjectCfg(
            prim_path=f"/World/Origin/Cable2{idx:03d}",
            spawn=spawn_cfg,
            init_state=CableObjectCfg.InitialStateCfg(pos=(cx, cy, cz), rot=z_axis_quat(angle)),
        )
        entities[f"Cable2{idx:03d}"] = CableObject(cfg=cfg)

    return entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, CableObject]):
    """Step the sim and periodically reset cable state."""
    sim_dt = sim.get_physics_dt()
    reset_steps = int(3.0 / sim_dt)
    count = 0

    while simulation_app.is_running():
        if count % reset_steps == 0:
            count = 0
            # Cables have no nodal snap-back; reset internal buffers only.
            for cable in entities.values():
                cable.reset()
            print("[INFO]: Resetting cable state...")
        sim.step()
        count += 1
        for cable in entities.values():
            cable.update(sim_dt)


def main():
    """Main entry point."""
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

    physics_cfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=20, rigid_body_contact_buffer_size=1024, rigid_contact_k_start=1.0e1, rigid_avbd_beta=1e2
        ),
        num_substeps=8,
    )
    # Soften body-body contact: lower ke + nonzero kd damps out the
    # spikes when many cable segments pile onto one segment. mu=1.0 keeps
    # cables from sliding off the pile.
    physics_cfg.model_cfg = NewtonModelCfg(
        shape_material_ke=1.0e3,
        shape_material_kd=1.0e0,
        shape_material_mu=1.0,
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=physics_cfg,
        visualizer_cfgs=[NewtonVisualizerCfg(eye=(2.5, 2.5, 1.0), lookat=(0.0, 0.0, 0.25))],
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
