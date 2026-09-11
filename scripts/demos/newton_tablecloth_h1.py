# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Have a Unitree H1 perform the tablecloth trick with Newton VBD and Newton IK.

The motion sequence uses Isaac Lab's GPU-resident Warp state-machine pattern.

.. code-block:: bash

    uv run python scripts/demos/newton_tablecloth_h1.py --device cuda:0
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton VBD H1 tablecloth trick.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--pull_speed", type=float, default=0.80, help="Task-space pull speed [m/s].")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import newton
import newton.utils
import numpy as np
import warp as wp
from isaaclab_newton.ik import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
    NewtonIKSolver,
    NewtonIKSolverCfg,
)
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
SUBSTEPS = 16
TABLE_TOP_Z = 1.09
RIGID_GAP = 0.001
PULL_DISTANCE = 0.40
PULL_RAMP_TIME = 0.25

SETTLE_END = wp.constant(0.50)
APPROACH_END = wp.constant(1.30)
DESCEND_END = wp.constant(1.90)
INSERT_END = wp.constant(2.40)
PRELIFT_END = wp.constant(3.00)
CLOSE_END = wp.constant(3.40)
LIFT_END = wp.constant(3.90)
PULL_START = wp.constant(4.50)

STATE_SETTLE = wp.constant(0)
STATE_APPROACH = wp.constant(1)
STATE_DESCEND = wp.constant(2)
STATE_INSERT = wp.constant(3)
STATE_PRELIFT = wp.constant(4)
STATE_CLOSE = wp.constant(5)
STATE_LIFT = wp.constant(6)
STATE_PINCH = wp.constant(7)
STATE_PULL = wp.constant(8)
STATE_HOLD = wp.constant(9)

GROUP_LEFT_THUMB = wp.constant(0)
GROUP_RIGHT_THUMB = wp.constant(1)
GROUP_LEFT_INDEX = wp.constant(2)
GROUP_RIGHT_INDEX = wp.constant(3)
GROUP_OTHER = wp.constant(4)

HAND_OFFSETS = ((0.146273, -0.068447, 0.028077), (0.148808, 0.068652, 0.026675))
HAND_ROTATIONS = ((-0.09, 0.46, 0.03, 0.88), (0.09023, 0.46115, -0.03008, 0.88221))
THUMB_CLOSED_VALUES = (
    (1.273907, 0.160957, 0.369535, 0.892908),
    (1.192278, 0.195421, 0.400690, 0.679765),
)


@configclass
class _H1VBDSolverCfg(VBDSolverCfg):
    """Unified AVBD/VBD settings matching the standalone Newton demo."""

    iterations: int = 10
    rigid_contact_hard: bool = False
    rigid_contact_history: bool = True
    rigid_body_contact_buffer_size: int = 512
    rigid_body_particle_contact_buffer_size: int = 8192
    rigid_joint_linear_ke: float = 1.0e6
    rigid_joint_angular_ke: float = 1.0e6
    rigid_joint_linear_kd: float = 1.0e2
    rigid_joint_angular_kd: float = 1.0e2


@configclass
class _H1CollisionCfg(NewtonCollisionPipelineCfg):
    """H1 full-surface soft contact and rigid correspondence settings."""

    broad_phase: str = "sap"
    soft_contact_margin: float = 0.008
    enable_rigid_soft_full_surface_contact: bool = True
    contact_matching: str = "latest"


@wp.func
def _smoothstep(value: float) -> float:
    u = wp.clamp(value, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@wp.kernel
def _infer_state_machine(
    sim_time: float,
    dt: float,
    pull_speed: float,
    keyframes: wp.array(dtype=wp.vec3),
    left_target: wp.array(dtype=wp.vec3),
    right_target: wp.array(dtype=wp.vec3),
    finger_fractions: wp.array(dtype=float),
    state: wp.array(dtype=wp.int32),
    pull_distance: wp.array(dtype=float),
):
    """Advance the single H1 task-space state machine entirely in Warp."""
    left = keyframes[0]
    right = keyframes[1]
    left_thumb = 0.0
    right_thumb = 0.0
    left_index = 0.0
    right_index = 0.0
    other = 0.0
    next_state = STATE_SETTLE

    if sim_time < SETTLE_END:
        pass
    elif sim_time < APPROACH_END:
        next_state = STATE_APPROACH
        u = _smoothstep((sim_time - SETTLE_END) / (APPROACH_END - SETTLE_END))
        left = wp.lerp(keyframes[0], keyframes[2], u)
        right = wp.lerp(keyframes[1], keyframes[3], u)
        other = 0.80 * u
    elif sim_time < DESCEND_END:
        next_state = STATE_DESCEND
        u = _smoothstep((sim_time - APPROACH_END) / (DESCEND_END - APPROACH_END))
        left = wp.lerp(keyframes[2], keyframes[4], u)
        right = wp.lerp(keyframes[3], keyframes[5], u)
        left_index = 0.75 * u
        right_index = 0.75 * u
        other = 0.80
    elif sim_time < INSERT_END:
        next_state = STATE_INSERT
        u = _smoothstep((sim_time - DESCEND_END) / (INSERT_END - DESCEND_END))
        left = wp.lerp(keyframes[4], keyframes[6], u)
        right = wp.lerp(keyframes[5], keyframes[7], u)
        left_index = 0.75
        right_index = 0.75
        other = 0.80
    elif sim_time < PRELIFT_END:
        next_state = STATE_PRELIFT
        u = _smoothstep((sim_time - INSERT_END) / (PRELIFT_END - INSERT_END))
        left = wp.lerp(keyframes[6], keyframes[8], u)
        right = wp.lerp(keyframes[7], keyframes[9], u)
        left_index = wp.lerp(0.75, 0.737080, u)
        right_index = wp.lerp(0.75, 0.713855, u)
        other = 0.80
    elif sim_time < CLOSE_END:
        next_state = STATE_CLOSE
        left = keyframes[8]
        right = keyframes[9]
        u = _smoothstep((sim_time - PRELIFT_END) / (CLOSE_END - PRELIFT_END))
        left_thumb = u
        right_thumb = u
        left_index = 0.737080
        right_index = 0.713855
        other = 0.80
    elif sim_time < LIFT_END:
        next_state = STATE_LIFT
        u = _smoothstep((sim_time - CLOSE_END) / (LIFT_END - CLOSE_END))
        left = wp.lerp(keyframes[8], keyframes[10], u)
        right = wp.lerp(keyframes[9], keyframes[11], u)
        left_thumb = 1.0
        right_thumb = 1.0
        left_index = 0.737080
        right_index = 0.713855
        other = 0.80
    elif sim_time < PULL_START:
        next_state = STATE_PINCH
        left = keyframes[10]
        right = keyframes[11]
        left_thumb = 1.0
        right_thumb = 1.0
        left_index = 0.737080
        right_index = 0.713855
        other = 0.80
    else:
        speed_ramp = _smoothstep((sim_time - PULL_START) / PULL_RAMP_TIME)
        distance = wp.min(pull_distance[0] + speed_ramp * pull_speed * dt, PULL_DISTANCE)
        pull_distance[0] = distance
        drop = 0.0
        if distance > 0.08:
            drop = 0.175 * _smoothstep((distance - 0.08) / (PULL_DISTANCE - 0.08))
        offset = wp.vec3(-distance, 0.0, -drop)
        left = keyframes[10] + offset
        right = keyframes[11] + offset
        left_thumb = 1.0
        right_thumb = 1.0
        left_index = 0.737080
        right_index = 0.713855
        other = 0.80
        next_state = STATE_PULL if distance < PULL_DISTANCE else STATE_HOLD

    left_target[0] = left
    right_target[0] = right
    finger_fractions[0] = left_thumb
    finger_fractions[1] = right_thumb
    finger_fractions[2] = left_index
    finger_fractions[3] = right_index
    finger_fractions[4] = other
    state[0] = next_state


@wp.kernel
def _set_finger_targets(
    joint_q: wp.array(dtype=float),
    finger_indices: wp.array(dtype=wp.int32),
    closed_values: wp.array(dtype=float),
    finger_groups: wp.array(dtype=wp.int32),
    fractions: wp.array(dtype=float),
):
    i = wp.tid()
    joint_q[finger_indices[i]] = fractions[finger_groups[i]] * closed_values[i]


@wp.kernel
def _update_control_targets(
    desired_q: wp.array(dtype=float),
    previous_q: wp.array(dtype=float),
    inv_dt: float,
    target_q: wp.array(dtype=float),
    target_qd: wp.array(dtype=float),
):
    i = wp.tid()
    delta = wp.clamp(desired_q[i] - previous_q[i], -40.0 / inv_dt, 40.0 / inv_dt)
    target_q[i] = previous_q[i] + delta
    target_qd[i] = delta * inv_dt
    previous_q[i] = target_q[i]


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [index for index, label in enumerate(labels) if label.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def _unit_quat(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    values_np = np.asarray(values, dtype=np.float32)
    values_np /= np.linalg.norm(values_np)
    return tuple(float(value) for value in values_np)


def _add_h1(builder: newton.ModelBuilder) -> tuple[dict[str, int], list[int]]:
    body_start = builder.body_count
    joint_start = builder.joint_count
    dof_start = builder.joint_dof_count
    shape_start = builder.shape_count
    sdf_defaults = (
        builder.default_shape_cfg.force_sdf,
        builder.default_shape_cfg.sdf_max_resolution,
        builder.default_shape_cfg.sdf_target_voxel_size,
    )
    builder.default_shape_cfg.configure_sdf(max_resolution=64)
    try:
        builder.add_mjcf(
            newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml",
            xform=wp.transform(wp.vec3(-0.75, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            ctrl_direct=False,
            parse_visuals=True,
            parse_sites=True,
            collider_classes=("collision",),
            no_class_as_colliders=True,
        )
    finally:
        (
            builder.default_shape_cfg.force_sdf,
            builder.default_shape_cfg.sdf_max_resolution,
            builder.default_shape_cfg.sdf_target_voxel_size,
        ) = sdf_defaults

    for dof in range(dof_start, builder.joint_dof_count):
        builder.joint_target_ke[dof] = 5.0e4
        builder.joint_target_kd[dof] = 5.0e2
    torso_dof = builder.joint_qd_start[_find_suffix(builder.joint_label, "torso_joint")]
    builder.joint_target_ke[torso_dof] = 2.0e5
    builder.joint_target_kd[torso_dof] = 2.0e3
    finger_tokens = tuple(
        f"/{side}_{digit}_" for side in ("L", "R") for digit in ("thumb", "index", "middle", "ring", "pinky")
    )
    for joint in range(joint_start, builder.joint_count):
        if any(token in builder.joint_label[joint] for token in finger_tokens):
            dof = builder.joint_qd_start[joint]
            builder.joint_target_ke[dof] = 4.0e4
            builder.joint_target_kd[dof] = 1.0e2

    bodies = {
        "torso": _find_suffix(builder.body_label, "torso_link"),
        "left_hand": _find_suffix(builder.body_label, "left_hand_link"),
        "right_hand": _find_suffix(builder.body_label, "right_hand_link"),
    }
    shape_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    grasp_tokens = ("/L_thumb_", "/L_index_", "/R_thumb_", "/R_index_")
    grasp_bodies = {
        body
        for body in range(body_start, builder.body_count)
        if any(token in builder.body_label[body] for token in grasp_tokens)
    }
    robot_shapes = []
    grasp_shape_count = 0
    for shape in range(shape_start, builder.shape_count):
        if not int(builder.shape_flags[shape]) & shape_flag:
            continue
        builder.shape_flags[shape] |= particle_flag
        builder.shape_gap[shape] = RIGID_GAP
        builder.shape_material_ke[shape] = 1.0e3
        builder.shape_material_kd[shape] = 1.0e-2
        builder.shape_material_mu[shape] = 0.50
        builder.shape_margin[shape] = 0.002
        builder.shape_sdf_padding[shape] = 0.012
        builder.shape_sdf_max_resolution[shape] = 64
        builder.shape_sdf_target_voxel_size[shape] = None
        robot_shapes.append(shape)
        if builder.shape_body[shape] in grasp_bodies:
            builder.shape_material_ke[shape] = 8.0e3
            builder.shape_material_kd[shape] = 2.0e1
            builder.shape_material_mu[shape] = 200.0
            builder.shape_sdf_max_resolution[shape] = 128
            grasp_shape_count += 1
    if grasp_shape_count != 12:
        raise RuntimeError(f"Expected 12 H1 thumb/index colliders, found {grasp_shape_count}")
    return bodies, robot_shapes


def _add_table(builder: newton.ModelBuilder) -> None:
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e1, mu=0.25, gap=RIGID_GAP, has_particle_collision=True)
    color = (0.46, 0.24, 0.10)
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, TABLE_TOP_Z - 0.04), wp.quat_identity()),
        hx=0.20,
        hy=0.36,
        hz=0.04,
        cfg=cfg,
        color=color,
    )
    for x_sign, y_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(x_sign * 0.15, y_sign * 0.30, 0.505), wp.quat_identity()),
            hx=0.03,
            hy=0.03,
            hz=0.505,
            cfg=cfg,
            color=color,
        )


def _add_cloth_and_tableware(builder: newton.ModelBuilder) -> None:
    resolution = 24
    width = 0.46
    depth = 0.70
    cell_x = width / resolution
    dim_y = round(depth / cell_x)
    cell_y = depth / dim_y
    cloth_z = TABLE_TOP_Z + 0.003
    builder.add_cloth_grid(
        pos=wp.vec3(-0.24, -0.5 * depth, cloth_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=resolution,
        dim_y=dim_y,
        cell_x=cell_x,
        cell_y=cell_y,
        mass=0.24 * width * depth / ((resolution + 1) * (dim_y + 1)),
        tri_ke=5.0e4,
        tri_ka=5.0e4,
        tri_kd=5.0e1,
        edge_ke=0.10,
        edge_kd=1.0e-3,
        particle_radius=0.001,
    )
    object_cfg = newton.ModelBuilder.ShapeConfig(
        density=2400.0,
        ke=1.0e4,
        kd=1.0e1,
        mu=0.04,
        gap=RIGID_GAP,
        has_particle_collision=True,
    )
    plate = builder.add_body(xform=wp.transform(wp.vec3(0.045, -0.08, cloth_z + 0.008), wp.quat_identity()))
    builder.add_shape_cylinder(plate, radius=0.065, half_height=0.008, cfg=object_cfg, color=(0.92, 0.91, 0.82))
    glass = builder.add_body(xform=wp.transform(wp.vec3(0.04, 0.11, cloth_z + 0.045), wp.quat_identity()))
    glass_cfg = object_cfg.copy()
    glass_cfg.density = 2500.0
    builder.add_shape_cylinder(glass, radius=0.032, half_height=0.045, cfg=glass_cfg, color=(0.52, 0.78, 0.90))
    fork = builder.add_body(xform=wp.transform(wp.vec3(0.08, -0.20, cloth_z + 0.004), wp.quat_identity()))
    fork_cfg = object_cfg.copy()
    fork_cfg.density = 8000.0
    builder.add_shape_box(fork, hx=0.060, hy=0.009, hz=0.004, cfg=fork_cfg, color=(0.72, 0.74, 0.76))


def _finger_data(model: newton.Model) -> tuple[list[int], list[float], list[int]]:
    q_starts = model.joint_q_start.numpy()
    indices = []
    values = []
    groups = []
    for side_index, side in enumerate(("L", "R")):
        thumb = THUMB_CLOSED_VALUES[side_index]
        entries = (
            ("thumb_proximal_yaw_joint", thumb[0]),
            ("thumb_proximal_pitch_joint", thumb[1]),
            ("thumb_intermediate_joint", thumb[2]),
            ("thumb_distal_joint", thumb[3]),
            ("index_proximal_joint", 1.2),
            ("index_intermediate_joint", 1.2),
            ("middle_proximal_joint", 1.0),
            ("middle_intermediate_joint", 1.0),
            ("ring_proximal_joint", 1.0),
            ("ring_intermediate_joint", 1.0),
            ("pinky_proximal_joint", 1.0),
            ("pinky_intermediate_joint", 1.0),
        )
        for suffix, value in entries:
            indices.append(int(q_starts[_find_suffix(model.joint_label, f"{side}_{suffix}")]))
            values.append(value)
            if suffix.startswith("thumb_"):
                groups.append(GROUP_LEFT_THUMB if side == "L" else GROUP_RIGHT_THUMB)
            elif suffix.startswith("index_"):
                groups.append(GROUP_LEFT_INDEX if side == "L" else GROUP_RIGHT_INDEX)
            else:
                groups.append(GROUP_OTHER)
    return indices, values, groups


def _make_ik(model: newton.Model, bodies: dict[str, int]) -> NewtonIKSolver:
    objectives = [
        NewtonIKPoseObjectiveCfg(
            body_name="left_hand",
            body_offset_pos=HAND_OFFSETS[0],
            use_relative_mode=False,
            position_weight=5.0,
            rotation_weight=0.2,
        ),
        NewtonIKPoseObjectiveCfg(
            body_name="right_hand",
            body_offset_pos=HAND_OFFSETS[1],
            use_relative_mode=False,
            position_weight=5.0,
            rotation_weight=0.2,
        ),
        NewtonIKPoseObjectiveCfg(
            body_name="torso",
            use_relative_mode=False,
            position_weight=50.0,
            rotation_weight=50.0,
        ),
        NewtonIKJointLimitObjectiveCfg(weight=1.0),
    ]
    return NewtonIKSolver(
        NewtonIKSolverCfg(iterations=24, lambda_initial=0.1),
        model=model,
        num_envs=1,
        device=str(model.device),
        objectives=objectives,
        link_resolver=bodies.__getitem__,
    )


def main() -> None:
    """Launch the H1 tablecloth demo."""
    if not np.isfinite(args_cli.pull_speed) or args_cli.pull_speed <= 0.0:
        raise ValueError("--pull_speed must be finite and positive")
    newton.use_coord_layout_targets = True
    physics_cfg = NewtonCfg(
        num_substeps=SUBSTEPS,
        collision_decimation=1,
        # Newton 1.5 allocates VBD contact history lazily, outside graph-safe initialization.
        use_cuda_graph=False,
        default_shape_cfg=NewtonShapeCfg(gap=RIGID_GAP, ke=1.0e4, kd=1.0e1, mu=0.25),
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=1.0e3, soft_contact_kd=1.0e-2, soft_contact_mu=0.25),
        collision_cfg=_H1CollisionCfg(),
        solver_cfg=_H1VBDSolverCfg(),
    )
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / FPS, device=args_cli.device, physics=resolved_physics_cfg)
        )
        sim.set_camera_view(eye=(-1.65, -1.80, 1.58), target=(-0.08, 0.0, 1.05))
        builder = NewtonManager.create_builder(gravity=(0.0, 0.0, -9.81))
        bodies, robot_shapes = _add_h1(builder)
        robot_coord_count = builder.joint_coord_count
        _add_table(builder)
        _add_cloth_and_tableware(builder)
        ground = builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e1, mu=0.25, gap=RIGID_GAP)
        )
        for shape in robot_shapes:
            builder.add_shape_collision_filter_pair(shape, ground)
        builder.color(include_bending=True)
        NewtonManager.set_builder(builder)
        sim.reset()

        model = NewtonManager.get_model()
        device = model.device
        ik_solver = _make_ik(model, bodies)
        left_objective = ik_solver.objectives_by_name["left_hand"]
        right_objective = ik_solver.objectives_by_name["right_hand"]
        torso_objective = ik_solver.objectives_by_name["torso"]
        initial_body_q = NewtonManager.get_state_0().body_q.numpy()
        torso_objective.position_objective.set_target_position(0, wp.vec3(*initial_body_q[bodies["torso"], :3]))
        torso_objective.rotation_objective.set_target_rotation(0, wp.quat(*initial_body_q[bodies["torso"], 3:]))
        left_objective.rotation_objective.set_target_rotation(0, wp.quat(*_unit_quat(HAND_ROTATIONS[0])))
        right_objective.rotation_objective.set_target_rotation(0, wp.quat(*_unit_quat(HAND_ROTATIONS[1])))

        grasp_y = 0.24
        keyframes = wp.array(
            [
                (-0.48, grasp_y, 1.24),
                (-0.48, -grasp_y, 1.24),
                (-0.30, grasp_y, 1.16),
                (-0.30, -grasp_y, 1.16),
                (-0.30, grasp_y, 1.050),
                (-0.30, -grasp_y, 1.052),
                (-0.225, grasp_y, 1.050),
                (-0.225, -grasp_y, 1.052),
                (-0.195, grasp_y, 1.110),
                (-0.195, -grasp_y, 1.110),
                (-0.195, grasp_y, 1.115),
                (-0.195, -grasp_y, 1.115),
            ],
            dtype=wp.vec3,
            device=device,
        )
        state = wp.zeros(1, dtype=wp.int32, device=device)
        pull_distance = wp.zeros(1, dtype=float, device=device)
        fractions = wp.zeros(5, dtype=float, device=device)
        finger_indices, closed_values, finger_groups = _finger_data(model)
        finger_indices_wp = wp.array(finger_indices, dtype=wp.int32, device=device)
        closed_values_wp = wp.array(closed_values, dtype=float, device=device)
        finger_groups_wp = wp.array(finger_groups, dtype=wp.int32, device=device)
        ik_seed = wp.clone(model.joint_q).reshape((1, model.joint_coord_count))
        control = NewtonManager.get_control()

        # Start from the solved rest pose instead of spending the settle phase moving out of the MJCF default.
        left_objective.position_objective.set_target_position(0, wp.vec3(-0.48, grasp_y, 1.24))
        right_objective.position_objective.set_target_position(0, wp.vec3(-0.48, -grasp_y, 1.24))
        ik_solver.cfg.iterations = 48
        solved = ik_solver.solve(ik_seed)
        solved_flat = solved.reshape((-1,))
        wp.copy(model.joint_q, solved_flat)
        for newton_state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
            newton.eval_fk(model, solved_flat, model.joint_qd, newton_state)
            newton_state.body_qd.zero_()
        previous_targets = wp.clone(solved_flat[:robot_coord_count])
        wp.copy(control.joint_target_q, previous_targets, count=robot_coord_count)
        control.joint_target_qd.zero_()
        ik_solver.cfg.iterations = 24
        wp.copy(ik_seed, solved)

        print("[INFO]: Setup complete. H1 Newton VBD tablecloth state machine is ready.", flush=True)
        step = 0
        while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
            wp.launch(
                _infer_state_machine,
                dim=1,
                inputs=[
                    step / FPS,
                    1.0 / FPS,
                    args_cli.pull_speed,
                    keyframes,
                    left_objective.position_objective.target_positions,
                    right_objective.position_objective.target_positions,
                    fractions,
                    state,
                    pull_distance,
                ],
            )
            solved = ik_solver.solve(ik_seed)
            solved_flat = solved.reshape((-1,))
            wp.launch(
                _set_finger_targets,
                dim=len(finger_indices),
                inputs=[solved_flat, finger_indices_wp, closed_values_wp, finger_groups_wp, fractions],
            )
            wp.copy(ik_seed, solved)
            wp.launch(
                _update_control_targets,
                dim=robot_coord_count,
                inputs=[
                    solved_flat,
                    previous_targets,
                    float(FPS),
                    control.joint_target_q,
                    control.joint_target_qd,
                ],
            )
            sim.step()
            step += 1


if __name__ == "__main__":
    main()
