# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drag a cable handle to lift a load through a single 4:1 pulley system.

.. code-block:: bash

    uv run python scripts/demos/newton_viewer_block_and_tackle.py
"""

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton block-and-tackle viewer dragging demo.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import newton
import newton.utils
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager, NewtonShapeCfg, VBDSolverCfg

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

MECHANICAL_ADVANTAGE = 4
LOAD_MASS = 5.0
HANDLE_MASS = 0.5  # Compensated in the load mass; raises Newton's picking-force limit.

CABLE_RADIUS = 0.004
CABLE_SEGMENT_LENGTH = 0.02
CABLE_GAP = 0.5 * CABLE_RADIUS
PULLEY_RADIUS = 0.045
WRAP_RADIUS = PULLEY_RADIUS + 1.25 * CABLE_RADIUS
SHEAVE_SPACING = 5.5 * CABLE_RADIUS

BLOCK_X = 0.50
MOVING_Z = 0.72
FIXED_Z = 1.10
LEAD_X = BLOCK_X + 3.0 * WRAP_RADIUS
PULL_X = LEAD_X + WRAP_RADIUS
LOAD_CENTER = wp.vec3(BLOCK_X, 0.0, 0.49)
HANDLE_HALF_EXTENTS = (0.025, 0.025, 0.03)


@configclass
class _BlockAndTackleVBDSolverCfg(VBDSolverCfg):
    """VBD contact settings for this cable and pulley scene."""

    rigid_contact_hard: bool = False
    rigid_body_contact_buffer_size: int = 512


def _append_arc(points: list[wp.vec3], center: wp.vec3, start: float, end: float) -> None:
    """Append a clockwise pulley arc in the vertical XZ plane."""
    delta = (end - start + math.pi) % (2.0 * math.pi) - math.pi
    if delta > 0.0:
        delta -= 2.0 * math.pi
    count = max(3, math.ceil(abs(delta) * WRAP_RADIUS / CABLE_SEGMENT_LENGTH))
    for index in range(count + 1):
        angle = start + delta * index / count
        point = wp.vec3(
            float(center[0]) + WRAP_RADIUS * math.cos(angle),
            float(center[1]),
            float(center[2]) + WRAP_RADIUS * math.sin(angle),
        )
        if float(wp.length(point - points[-1])) > 1.0e-8:
            points.append(point)


def _resample_route(points: list[wp.vec3]) -> tuple[list[wp.vec3], float]:
    """Resample a route into equal-length cable segments."""
    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + float(wp.length(end - start)))
    segment_count = math.ceil(lengths[-1] / CABLE_SEGMENT_LENGTH)
    spacing = lengths[-1] / segment_count
    result = [points[0]]
    point_index = 1
    for segment_index in range(1, segment_count):
        distance = spacing * segment_index
        while lengths[point_index] < distance:
            point_index += 1
        alpha = (distance - lengths[point_index - 1]) / (lengths[point_index] - lengths[point_index - 1])
        result.append(points[point_index - 1] * (1.0 - alpha) + points[point_index] * alpha)
    result.append(points[-1])
    return result, spacing


def _cable_route(
    moving_centers: list[wp.vec3], fixed_centers: list[wp.vec3], lead_center: wp.vec3
) -> tuple[list[wp.vec3], float]:
    """Route one cable through the moving and fixed pulley blocks."""
    points = [
        wp.vec3(
            float(moving_centers[0][0]) + WRAP_RADIUS,
            float(moving_centers[0][1]),
            float(fixed_centers[0][2]) - 2.0 * WRAP_RADIUS,
        )
    ]
    for index, (moving_center, fixed_center) in enumerate(zip(moving_centers, fixed_centers, strict=True)):
        _append_arc(points, moving_center, 0.0, -math.pi)
        _append_arc(points, fixed_center, math.pi, 0.0 if index == 0 else 0.5 * math.pi)
    _append_arc(points, lead_center, 0.5 * math.pi, 0.0)
    points.append(wp.vec3(PULL_X, float(lead_center[1]), 0.50))
    return _resample_route(points)


def _add_box(
    builder: newton.ModelBuilder,
    body: int,
    center: wp.vec3,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    density: float = 0.0,
    collision: bool = False,
) -> None:
    """Add a compact visual or colliding box."""
    builder.add_shape_box(
        body=body,
        xform=wp.transform(center, wp.quat_identity()),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=newton.ModelBuilder.ShapeConfig(
            density=density,
            ke=1.0e5,
            kd=20.0,
            mu=0.8,
            has_shape_collision=collision,
            has_particle_collision=collision,
        ),
        color=color,
    )


def _add_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    parent: int,
    parent_origin: wp.vec3,
    color: tuple[float, float, float],
) -> int:
    """Add one passive grooved sheave and return its revolute joint."""
    body = builder.add_link(xform=wp.transform(center, wp.quat_identity()))
    joint = builder.add_joint_revolute(
        parent=parent,
        child=body,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(center - parent_origin, wp.quat_identity()),
        armature=1.0e-4,
        friction=0.0,
    )
    align_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    groove_half_width = 1.55 * CABLE_RADIUS
    flange_half_width = 0.6 * CABLE_RADIUS
    for y, radius, half_width, shade, friction in (
        (0.0, PULLEY_RADIUS, groove_half_width, color, 0.1),
        (
            -(groove_half_width + flange_half_width),
            PULLEY_RADIUS + 3.2 * CABLE_RADIUS,
            flange_half_width,
            tuple(0.68 * value for value in color),
            0.05,
        ),
        (
            groove_half_width + flange_half_width,
            PULLEY_RADIUS + 3.2 * CABLE_RADIUS,
            flange_half_width,
            tuple(0.68 * value for value in color),
            0.05,
        ),
    ):
        builder.add_shape_cylinder(
            body=body,
            xform=wp.transform(wp.vec3(0.0, y, 0.0), align_to_y),
            radius=radius,
            half_height=half_width,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.1, ke=1.0e5, kd=0.0, mu=friction),
            color=shade,
        )
    return joint


def _build_system(builder: newton.ModelBuilder) -> tuple[int, list[wp.transform]]:
    """Build one complete 4:1 block-and-tackle system."""
    sheave_y = (-0.5 * SHEAVE_SPACING, 0.5 * SHEAVE_SPACING)
    moving_centers = [wp.vec3(BLOCK_X, y, MOVING_Z) for y in sheave_y]
    fixed_centers = [wp.vec3(BLOCK_X, y, FIXED_Z) for y in sheave_y]
    lead_center = wp.vec3(LEAD_X, sheave_y[-1], FIXED_Z)

    _add_box(builder, -1, wp.vec3(0.5 * (BLOCK_X + PULL_X), 0.0, 1.19), (0.18, 0.07, 0.025), (0.16, 0.22, 0.30))
    _add_box(builder, -1, wp.vec3(BLOCK_X, 0.0, 1.145), (0.025, 0.065, 0.045), (0.16, 0.22, 0.30))
    _add_box(builder, -1, wp.vec3(LEAD_X, sheave_y[-1], 1.145), (0.025, 0.025, 0.045), (0.16, 0.22, 0.30))
    _add_box(builder, -1, wp.vec3(BLOCK_X, 0.0, 0.39), (0.14, 0.12, 0.01), (0.24, 0.27, 0.30), collision=True)

    load_body = builder.add_link(xform=wp.transform(LOAD_CENTER, wp.quat_identity()), label="load")
    load_half_extents = (0.085, 0.08, 0.09)
    load_volume = 8.0 * math.prod(load_half_extents)
    _add_box(
        builder,
        load_body,
        wp.vec3(0.0),
        load_half_extents,
        (0.72, 0.16, 0.12),
        density=(LOAD_MASS + HANDLE_MASS * MECHANICAL_ADVANTAGE) / load_volume,
        collision=True,
    )
    _add_box(builder, load_body, wp.vec3(0.0, 0.0, 0.16), (0.025, 0.065, 0.07), (0.50, 0.18, 0.12))
    load_joint = builder.add_joint_prismatic(
        parent=-1,
        child=load_body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(LOAD_CENTER, wp.quat_identity()),
        target_kd=20.0,
    )

    moving_joints = [
        _add_pulley(builder, center, load_body, LOAD_CENTER, (0.88, 0.54, 0.12)) for center in moving_centers
    ]
    fixed_joints = [_add_pulley(builder, center, -1, wp.vec3(0.0), (0.12, 0.38, 0.78)) for center in fixed_centers]
    lead_joint = _add_pulley(builder, lead_center, -1, wp.vec3(0.0), (0.12, 0.38, 0.78))
    builder.add_articulation([load_joint, *moving_joints], label="moving_block")
    for joint in [*fixed_joints, lead_joint]:
        builder.add_articulation([joint])

    route, segment_length = _cable_route(moving_centers, fixed_centers, lead_center)
    route_quaternions = newton.utils.create_parallel_transport_cable_quaternions(route)
    cable_segment_count = len(route) - 1
    straight_points, straight_quaternions = newton.utils.create_straight_cable_points_and_quaternions(
        start=route[0],
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=cable_segment_count * segment_length,
        num_segments=cable_segment_count,
    )
    cable_bodies, cable_joints = builder.add_rod(
        positions=straight_points,
        quaternions=straight_quaternions,
        radius=CABLE_RADIUS,
        body_frame_origin="com",
        cfg=newton.ModelBuilder.ShapeConfig(density=10.0, ke=1.0e5, kd=0.0, mu=0.1, gap=CABLE_GAP),
        wrap_in_articulation=False,
        color=(0.82, 0.72, 0.46),
        label="block_and_tackle_cable",
    )
    endpoint = 0.5 * segment_length
    anchor_joint = builder.add_joint_ball(
        parent=-1,
        child=cable_bodies[0],
        parent_xform=wp.transform(route[0], wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -endpoint), wp.quat_identity()),
        label="cable_anchor",
    )
    handle_volume = 8.0 * math.prod(HANDLE_HALF_EXTENTS)
    _add_box(
        builder,
        cable_bodies[-1],
        wp.vec3(0.0, 0.0, endpoint + HANDLE_HALF_EXTENTS[2]),
        HANDLE_HALF_EXTENTS,
        (0.96, 0.78, 0.26),
        density=HANDLE_MASS / handle_volume,
        collision=True,
    )
    builder.add_articulation([*cable_joints, anchor_joint], label="cable")

    for index, body_a in enumerate(cable_bodies):
        for body_b in cable_bodies[index + 1 :]:
            for shape_a in builder.body_shapes[body_a]:
                for shape_b in builder.body_shapes[body_b]:
                    builder.add_shape_collision_filter_pair(shape_a, shape_b)

    wrapped_xforms = [
        wp.transform(route[index] + 0.5 * (route[index + 1] - route[index]), route_quaternions[index])
        for index in range(cable_segment_count)
    ]
    return cable_bodies[0], wrapped_xforms


def _initialize_wrapped_cable(cable_body_start: int, wrapped_xforms: list[wp.transform]) -> None:
    """Restore model defaults and wrap the structurally straight cable."""
    model = NewtonManager.get_model()
    wrapped = wp.array(wrapped_xforms, dtype=wp.transform, device=model.device)
    for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
        wp.copy(state.body_q, model.body_q)
        wp.copy(state.body_qd, model.body_qd)
        wp.copy(state.body_q, wrapped, dest_offset=cable_body_start, count=len(wrapped_xforms))
        state.body_f.zero_()
    NewtonManager._solver.reset(NewtonManager.get_state_0(), flags=0)


def run_simulator(sim: sim_utils.SimulationContext) -> None:
    """Run until the viewer closes or the optional step limit is reached."""
    step_count = 0
    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
        sim.step()
        step_count += 1


def main() -> None:
    """Launch the block-and-tackle dragging demo."""
    physics_cfg = NewtonCfg(
        num_substeps=16,
        collision_decimation=1,
        default_shape_cfg=NewtonShapeCfg(gap=CABLE_GAP, ke=1.0e5, kd=20.0, mu=0.5),
        solver_cfg=_BlockAndTackleVBDSolverCfg(),
    )
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, physics=resolved_physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(1.35, -2.1, 1.25), target=(0.50, 0.0, 0.72))
        builder = NewtonManager.create_builder()
        builder.rigid_gap = CABLE_GAP
        cable_body_start, wrapped_xforms = _build_system(builder)
        builder.color(balance_colors=False)
        NewtonManager.set_builder(builder)
        sim.reset()
        _initialize_wrapped_cable(cable_body_start, wrapped_xforms)
        print("[INFO]: Setup complete. Right-drag the yellow cable handle downward to lift the red load.", flush=True)
        run_simulator(sim)


if __name__ == "__main__":
    main()
