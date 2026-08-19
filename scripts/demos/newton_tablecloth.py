# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the classic tablecloth trick at five pull speeds with Newton VBD.

.. code-block:: bash

    uv run python scripts/demos/newton_tablecloth.py --device cuda:0
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Compare five Newton VBD tablecloth pull speeds.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

FPS = 60
SUBSTEPS = 25
SETTLE_TIME = 0.5
PULL_DISTANCE = 1.25
# Newton 1.5 needs a slightly sharper final pull than Newton main to retain the glass.
PULL_SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0)

TABLE_HALF_WIDTH = 0.50
TABLE_HALF_DEPTH = 0.36
TABLE_TOP_Z = 0.75
TABLETOP_HALF_HEIGHT = 0.04
LANE_SPACING = 0.95

CLOTH_WIDTH = 1.08
CLOTH_DEPTH = 0.70
CLOTH_RESOLUTION = 24
CLOTH_PARTICLE_RADIUS = 0.0005


@configclass
class _TableclothVBDSolverCfg(VBDSolverCfg):
    """VBD options required by the rigid-soft tablecloth scene."""

    iterations: int = 15
    rigid_contact_hard: bool = False
    rigid_body_contact_buffer_size: int = 512
    rigid_body_particle_contact_buffer_size: int = 8192


@configclass
class _TableclothCollisionCfg(NewtonCollisionPipelineCfg):
    """Full-surface collision with stable resting-contact correspondence."""

    broad_phase: str = "nxn"
    soft_contact_margin: float = 0.005
    enable_rigid_soft_full_surface_contact: bool = True
    contact_matching: str = "sticky"


@wp.kernel
def _advance_time(elapsed: wp.array(dtype=float), dt: float):
    elapsed[0] += dt


@wp.kernel
def _advance_pull(
    distance: wp.array(dtype=float), speed: wp.array(dtype=float), elapsed: wp.array(dtype=float), dt: float
):
    lane = wp.tid()
    if elapsed[0] >= SETTLE_TIME:
        distance[lane] = wp.min(distance[lane] + speed[lane] * dt, PULL_DISTANCE)


@wp.kernel
def _move_pulled_edge(
    particle_indices: wp.array(dtype=wp.int32),
    lane_indices: wp.array(dtype=wp.int32),
    rest_positions: wp.array(dtype=wp.vec3),
    pull_speeds: wp.array(dtype=float),
    pull_distances: wp.array(dtype=float),
    elapsed: wp.array(dtype=float),
    particle_q_0: wp.array(dtype=wp.vec3),
    particle_q_1: wp.array(dtype=wp.vec3),
    particle_qd_0: wp.array(dtype=wp.vec3),
    particle_qd_1: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    lane = lane_indices[i]
    distance = pull_distances[lane]
    descent = 0.0
    descent_velocity = 0.0
    if distance > 0.10:
        descent = 0.30 * wp.clamp((distance - 0.10) / (PULL_DISTANCE - 0.10), 0.0, 1.0)
    velocity = 0.0
    if elapsed[0] >= SETTLE_TIME and distance < PULL_DISTANCE:
        velocity = pull_speeds[lane]
        if distance > 0.10:
            descent_velocity = 0.30 * pull_speeds[lane] / (PULL_DISTANCE - 0.10)
    position = rest_positions[i] + wp.vec3(distance, 0.0, -descent)
    linear_velocity = wp.vec3(velocity, 0.0, -descent_velocity)
    index = particle_indices[i]
    particle_q_0[index] = position
    particle_q_1[index] = position
    particle_qd_0[index] = linear_velocity
    particle_qd_1[index] = linear_velocity


def _collision_cfgs(density: float, friction: float) -> tuple[newton.ModelBuilder.ShapeConfig, ...]:
    """Create separate rigid-support and cloth-contact representations."""
    rigid = newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=1.0e5,
        kd=1.0e2,
        mu=friction,
        margin=0.002,
        has_particle_collision=False,
    )
    cloth = rigid.copy()
    cloth.density = 0.0
    cloth.ke = 1.0e3
    cloth.kd = 1.0e1
    cloth.has_shape_collision = False
    cloth.has_particle_collision = True
    cloth.is_visible = False
    return rigid, cloth


def _add_table(builder: newton.ModelBuilder, lane_y: float) -> None:
    rigid_cfg, cloth_cfg = _collision_cfgs(0.0, 0.70)
    color = (0.46, 0.24, 0.10)
    tabletop = wp.transform(wp.vec3(0.0, lane_y, TABLE_TOP_Z - TABLETOP_HALF_HEIGHT), wp.quat_identity())
    for cfg in (rigid_cfg, cloth_cfg):
        builder.add_shape_box(
            -1,
            xform=tabletop,
            hx=TABLE_HALF_WIDTH,
            hy=TABLE_HALF_DEPTH,
            hz=TABLETOP_HALF_HEIGHT,
            cfg=cfg,
            color=color,
        )
    leg_half_height = 0.5 * (TABLE_TOP_Z - 2.0 * TABLETOP_HALF_HEIGHT)
    for x_sign, y_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        xform = wp.transform(
            wp.vec3(
                x_sign * (TABLE_HALF_WIDTH - 0.07),
                lane_y + y_sign * (TABLE_HALF_DEPTH - 0.07),
                leg_half_height,
            ),
            wp.quat_identity(),
        )
        for cfg in (rigid_cfg, cloth_cfg):
            builder.add_shape_box(-1, xform=xform, hx=0.035, hy=0.035, hz=leg_half_height, cfg=cfg, color=color)


def _add_tableware(builder: newton.ModelBuilder, lane_y: float, z_base: float) -> None:
    def add_cylinder(position, radius, half_height, density, color):
        body = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat_identity()))
        for cfg in _collision_cfgs(density, 0.70):
            builder.add_shape_cylinder(body, radius=radius, half_height=half_height, cfg=cfg, color=color)
        return body

    add_cylinder((-0.22, lane_y - 0.08, z_base + 0.012), 0.105, 0.012, 2400.0, (0.92, 0.91, 0.82))
    add_cylinder((0.02, lane_y + 0.12, z_base + 0.050), 0.050, 0.050, 2500.0, (0.52, 0.78, 0.90))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.22, lane_y - 0.11, z_base + 0.006), wp.quat_identity()))
    for cfg in _collision_cfgs(8000.0, 0.70):
        builder.add_shape_box(body, hx=0.110, hy=0.016, hz=0.006, cfg=cfg, color=(0.72, 0.74, 0.76))


def _build_scene(builder: newton.ModelBuilder) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lane_centers = (np.arange(len(PULL_SPEEDS), dtype=np.float32) - 0.5 * (len(PULL_SPEEDS) - 1)) * LANE_SPACING
    pulled_indices = []
    pulled_lanes = []
    pulled_rest = []
    dim_x = CLOTH_RESOLUTION
    cell_x = CLOTH_WIDTH / dim_x
    dim_y = round(CLOTH_DEPTH / cell_x)
    cell_y = CLOTH_DEPTH / dim_y
    particle_mass = 0.24 * CLOTH_WIDTH * CLOTH_DEPTH / ((dim_x + 1) * (dim_y + 1))
    cloth_z = TABLE_TOP_Z + CLOTH_PARTICLE_RADIUS + 0.002

    for lane, lane_y in enumerate(lane_centers):
        _add_table(builder, float(lane_y))
        start = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * CLOTH_WIDTH, float(lane_y - 0.5 * CLOTH_DEPTH), cloth_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=particle_mass,
            fix_right=True,
            tri_ke=5.0e4,
            tri_ka=5.0e4,
            tri_kd=5.0e1,
            edge_ke=0.10,
            edge_kd=1.0e-3,
            particle_radius=CLOTH_PARTICLE_RADIUS,
        )
        for y in range(dim_y + 1):
            index = start + y * (dim_x + 1) + dim_x
            pulled_indices.append(index)
            pulled_lanes.append(lane)
            pulled_rest.append(builder.particle_q[index])
        _add_tableware(builder, float(lane_y), cloth_z + CLOTH_PARTICLE_RADIUS)

    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0e2, mu=0.70))
    builder.color(include_bending=True)
    return (
        np.asarray(pulled_indices, dtype=np.int32),
        np.asarray(pulled_lanes, dtype=np.int32),
        np.asarray(pulled_rest, dtype=np.float32),
    )


def main() -> None:
    """Launch the five-speed tablecloth demo."""
    newton.use_coord_layout_targets = True
    physics_cfg = NewtonCfg(
        num_substeps=SUBSTEPS,
        collision_decimation=1,
        default_shape_cfg=NewtonShapeCfg(gap=0.001, ke=1.0e5, kd=1.0e2, mu=0.70),
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=1.0e5, soft_contact_kd=1.0e2, soft_contact_mu=0.10),
        collision_cfg=_TableclothCollisionCfg(),
        solver_cfg=_TableclothVBDSolverCfg(),
    )
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / FPS, device=args_cli.device, physics=resolved_physics_cfg)
        )
        sim.set_camera_view(eye=(2.8, -5.2, 2.4), target=(0.1, 0.0, 0.65))
        builder = NewtonManager.create_builder(gravity=(0.0, 0.0, -9.81))
        pulled_indices, pulled_lanes, pulled_rest = _build_scene(builder)
        NewtonManager.set_builder(builder)
        device = args_cli.device
        indices_wp = wp.array(pulled_indices, dtype=wp.int32, device=device)
        lanes_wp = wp.array(pulled_lanes, dtype=wp.int32, device=device)
        rest_wp = wp.array(pulled_rest, dtype=wp.vec3, device=device)
        speeds_wp = wp.array(PULL_SPEEDS, dtype=float, device=device)
        distances_wp = wp.zeros(len(PULL_SPEEDS), dtype=float, device=device)
        elapsed_wp = wp.zeros(1, dtype=float, device=device)

        def update_pulled_edge(_state: newton.State) -> None:
            """Advance and prescribe the pulled edge before every solver substep."""
            wp.launch(_advance_time, dim=1, inputs=[elapsed_wp, 1.0 / (FPS * SUBSTEPS)])
            wp.launch(
                _advance_pull,
                dim=len(PULL_SPEEDS),
                inputs=[distances_wp, speeds_wp, elapsed_wp, 1.0 / (FPS * SUBSTEPS)],
            )
            wp.launch(
                _move_pulled_edge,
                dim=len(pulled_indices),
                inputs=[
                    indices_wp,
                    lanes_wp,
                    rest_wp,
                    speeds_wp,
                    distances_wp,
                    elapsed_wp,
                    NewtonManager.get_state_0().particle_q,
                    NewtonManager.get_state_1().particle_q,
                    NewtonManager.get_state_0().particle_qd,
                    NewtonManager.get_state_1().particle_qd,
                ],
            )

        NewtonManager.register_state_force_callback(update_pulled_edge)
        sim.reset()
        print("[INFO]: Setup complete. Five Newton VBD tablecloth pulls are ready.", flush=True)

        step = 0
        while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
            sim.step()
            step += 1


if __name__ == "__main__":
    main()
