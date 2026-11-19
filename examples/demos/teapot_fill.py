# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fill and pour water from the Utah teapot with Newton implicit MPM.

The teapot is a hollow, double-walled shell, so the fluid is seeded in its enclosed air cavity with
:func:`~isaaclab.utils.warp.sample_particles_in_cavity` instead of using a winding-number volume fill.

.. code-block:: bash

    # Fast Newton visualizer (the default):
    uvx isaaclab demo teapot-fill --device cuda:0 --visualizer newton_gl
    # Display both the raw particles and reconstructed surface:
    uvx isaaclab demo teapot-fill --visualizer newton_gl --fluid_render_mode both
    # Fuller / coarser (faster) fill:
    uvx isaaclab demo teapot-fill --fill_level 1.0 --fill_spacing 0.003
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.app import launch_simulation
from isaaclab.examples.demos.teapot.scene import (
    CONTAINER_BASE_POS,
    RENDER_INTERVAL,
    TeapotFillSceneCfg,
    add_teapot_args,
    configure_scene,
    create_surface_renderer,
    particle_spacing,
)
from isaaclab.examples.demos.teapot.scene import (
    create_sim_cfg as create_mpm_sim_cfg,
)
from isaaclab.utils.assets import retrieve_file_path

if TYPE_CHECKING:
    from isaaclab.sim import SimulationCfg

logger = logging.getLogger(__name__)

HOLD_TIME = 0.55
TILT_TIME = 2.0
POUR_ANGLE = math.radians(65.0)
CONTAINER_LIFT_HEIGHT = 0.24
CONTAINER_LIFT_TIME = 3.0


def quat_y(angle_rad: float) -> tuple[float, float, float, float]:
    """Return an XYZW quaternion for a rotation about +Y."""
    half = 0.5 * angle_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def container_pose_at_time(
    sim_time: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float, float, float],
]:
    """Return teapot ``(position, orientation, twist)`` for the scripted pour."""
    raw = (sim_time - HOLD_TIME) / TILT_TIME
    clamped = max(0.0, min(1.0, raw))
    alpha = clamped * clamped * (3.0 - 2.0 * clamped)
    alpha_dot = (6.0 * clamped * (1.0 - clamped)) / TILT_TIME if 0.0 < raw < 1.0 else 0.0

    lift_raw = (sim_time - HOLD_TIME - TILT_TIME) / CONTAINER_LIFT_TIME
    lift_alpha = max(0.0, min(1.0, lift_raw))
    lift_speed = CONTAINER_LIFT_HEIGHT / CONTAINER_LIFT_TIME if 0.0 < lift_raw < 1.0 else 0.0

    angle = POUR_ANGLE * alpha
    angular_speed = POUR_ANGLE * alpha_dot
    pos = (
        CONTAINER_BASE_POS[0],
        CONTAINER_BASE_POS[1],
        CONTAINER_BASE_POS[2] + CONTAINER_LIFT_HEIGHT * lift_alpha,
    )
    # Newton spatial vectors are (linear, angular).
    twist = (0.0, 0.0, lift_speed, 0.0, angular_speed, 0.0)
    return pos, quat_y(angle), twist


def create_sim_cfg(args: argparse.Namespace | None = None) -> SimulationCfg:
    """Build the plain MPM demo configuration."""
    return create_mpm_sim_cfg(_parse_args() if args is None else args)


def main() -> None:
    """Run the kinematic teapot pour."""
    args = _parse_args()
    sim_cfg = create_sim_cfg(args)
    fill_spacing = particle_spacing(args)

    with launch_simulation(sim_cfg, args):
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(sim_cfg)
        scene_cfg = TeapotFillSceneCfg(num_envs=1, env_spacing=0.0)
        configure_scene(scene_cfg, args, retrieve_file_path(args.container_usd), fill_spacing)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        surface = create_surface_renderer(sim, args, fill_spacing)
        if surface is not None:
            surface.update()
        fluid = scene["fluid"]
        logger.info("Sampled %d fluid particles in the teapot.", fluid.num_instances * fluid.particles_per_object)

        container = scene["container"]
        dt = sim.get_physics_dt()
        step = 0
        while (args.max_steps < 0 or step < args.max_steps) and sim.is_running():
            position, orientation, twist = container_pose_at_time(step * dt)
            pose = torch.tensor([position + orientation], dtype=torch.float32, device=sim.device)
            velocity = torch.tensor([twist], dtype=torch.float32, device=sim.device)
            container.write_root_link_pose_to_sim_index(root_pose=pose)
            container.write_root_link_velocity_to_sim_index(root_velocity=velocity)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            if sim.is_rendering and step % RENDER_INTERVAL == 0:
                if surface is not None:
                    surface.update()
                sim.render()
            step += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
    add_teapot_args(parser, max_steps=6000, fluid_render_mode="surface")
    return parser.parse_args()


if __name__ == "__main__":
    main()
