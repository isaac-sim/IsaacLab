# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physically grasp the Utah teapot and pour MPM water during a 25 cm rise.

This experimental example uses a fixed index grasp and motion.
The complete sequence checks the retained water mass for the default catch bowl.

.. code-block:: bash

    # Newton GL:
    uv run isaaclab example rizon-sharpa-teapot

The pinned robot USD and textures are downloaded and verified on first use.
Set ``ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT`` to an existing bundle directory
to use it without downloading.
"""

from __future__ import annotations

import argparse
import logging

from isaaclab.app import launch_simulation
from isaaclab.examples.demos.teapot.grasp_scene import GraspedTeapotSceneCfg, make_solver_cfg, prepare_robot_usd
from isaaclab.examples.demos.teapot.motion import SEQUENCE_DURATION, TeapotPourMotion
from isaaclab.examples.demos.teapot.scene import (
    BOWL_LOCAL_Z_MAX,
    DEFAULT_BOWL_USD,
    RENDER_INTERVAL,
    SIMULATION_HZ,
    add_teapot_args,
    configure_scene,
    create_sim_cfg,
    create_surface_renderer,
    particle_spacing,
)
from isaaclab.utils.assets import retrieve_file_path

logger = logging.getLogger(__name__)
SEQUENCE_STEPS = round(SIMULATION_HZ * SEQUENCE_DURATION)
# A cylinder strictly inside the authored Bowl_G01 cavity [m].
BOWL_SAMPLE_RADIUS = 0.021
BOWL_FLOOR_HEIGHT = 0.0049
MIN_DELIVERED_MASS = 0.0005


def main() -> None:
    """Run the physical Sharpa pour."""
    parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
    add_teapot_args(parser, max_steps=SEQUENCE_STEPS, fluid_render_mode="particles")
    args = parser.parse_args()
    sim_cfg = create_sim_cfg(args)
    # Quadratic velocity interpolation and dense collider sampling keep the stream inside the bowl.
    sim_cfg.physics.solver_cfg = make_solver_cfg(
        sim_cfg.physics.solver_cfg.replace(
            air_drag=1.0,
            strain_basis="P1d",
            transfer_scheme="pic",
            velocity_basis="B2",
            max_iterations=25,
            max_active_cell_count=1 << 15,
            collider_basis="pic512",
        )
    )
    sim_cfg.visualizer_cfgs.eye = (0.55, -1.5, 1.72)
    sim_cfg.visualizer_cfgs.lookat = (-0.28, 0.0, 1.22)
    fill_spacing = particle_spacing(args)

    with launch_simulation(sim_cfg, args):
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(sim_cfg)
        container_usd = retrieve_file_path(args.container_usd)
        scene_cfg = GraspedTeapotSceneCfg(num_envs=1, env_spacing=0.0)
        scene_cfg.container.spawn.contact_source = container_usd
        scene_cfg.robot.spawn.usd_path = prepare_robot_usd()
        configure_scene(scene_cfg, args, container_usd, fill_spacing, shell_margin=0.0005)
        scene = InteractiveScene(scene_cfg)
        sim.reset()

        robot = scene["robot"]
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.warp)
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.warp)
        sim.forward()
        motion = TeapotPourMotion(
            robot,
            scene["container"],
            sim.backend_manager.backend.solver.solver("robot").mj_model,
            sim_dt=sim.get_physics_dt(),
        )
        robot.write_joint_position_to_sim_index(position=motion.initial_joint_positions)
        sim.forward()

        surface = create_surface_renderer(sim, args, fill_spacing)
        if surface is not None:
            surface.update()
        fluid = scene["fluid"]
        logger.info("Sampled %d fluid particles in the teapot.", fluid.num_instances * fluid.particles_per_object)

        dt = sim.get_physics_dt()
        step = 0
        while (args.max_steps < 0 or step < args.max_steps) and sim.is_running():
            motion.update(step * dt)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            if sim.is_rendering and step % RENDER_INTERVAL == 0:
                if surface is not None:
                    surface.update()
                sim.render()
            step += 1
            if step == SEQUENCE_STEPS and args.bowl_usd == DEFAULT_BOWL_USD:
                position = fluid.data.particle_pos_w.torch[0]
                center = position.new_tensor(scene_cfg.catch_bowl.init_state.pos)
                local_position = position - center
                in_bowl = (
                    (local_position[:, :2].square().sum(dim=-1) < BOWL_SAMPLE_RADIUS**2)
                    & (local_position[:, 2] > BOWL_FLOOR_HEIGHT)
                    & (local_position[:, 2] < BOWL_LOCAL_Z_MAX)
                )
                delivered_mass = in_bowl.sum().item() * scene_cfg.fluid.spawn.mass
                logger.info("Water retained in the bowl interior: %.2f g.", 1000.0 * delivered_mass)
                if delivered_mass < MIN_DELIVERED_MASS:
                    raise RuntimeError(
                        f"The bowl interior retained {1000.0 * delivered_mass:.2f} g of water. "
                        f"Expected at least {1000.0 * MIN_DELIVERED_MASS:.2f} g."
                    )


if __name__ == "__main__":
    main()
