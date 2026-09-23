# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the Newton browser demo bundles used by the documentation.

The browser runtime and compiler are provided by ``newton-web``. The exported
bundles are deployment assets; this script keeps the simulation definitions and
the locomotion policy contracts in Isaac Lab's source tree.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import newton
import numpy as np
import torch
import trimesh
import warp as wp
from mujoco_warp._src.types import DisableBit
from newton_web import Parameter, export_graph

STIFFNESS_DT = 1.0 / 240.0
CLOTH_DT = 1.0 / 120.0
G1_DOF = 29
G1_POLICY_SHA256 = "4c92c5a64d1220ab02b042e77bdd69bcd2c0310590755c3dd53b5bde229d26e4"
G1_VISUAL_URDF_SHA256 = "c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8"
G1_VISUAL_SOURCE_REVISION = "ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee"
G1_BROWSER_SOLVER_ITERATIONS = 1
G1_COLLISION_LINKS = {
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
}
ANYMAL_D_USD_SHA256 = "8b756c3690808b3b6a9a3fadc62ab843788c7b7835bbe085f0145f173521dc74"
ANYMAL_D_MESH_SHA256 = "a864b5b9e192592595490f4116319476090f6789830057854c6532020dfc3d33"
ANYMAL_D_CHECKPOINT_SHA256 = "0654295241696cdc7855f517a8d94a4951a243f6b21d73152225162ea01aeaaa"
ANYMAL_D_DOF = 12
ANYMAL_BROWSER_KP = 200.0
ANYMAL_BROWSER_KD = 20.0
CARTPOLE_USD_SHA256 = "c98ce5dbb174876998052d486036fb79e07f52320851526a4ff61c60ed2db043"
CARTPOLE_MESH_SHA256 = "27976d05b7ee47d7674ab540b8b692113c0052b36f76a5fe9deb01aadc392aa1"
CARTPOLE_CHECKPOINT_SHA256 = "251c836e5b6fb9b229ec5e542b7a9071ec49e2da3ebeb3dc230cc77259c76eef"


@wp.kernel
def _set_lame_parameters(
    shear: wp.array(dtype=float), volume: wp.array(dtype=float), materials: wp.array2d(dtype=float), first_tet: int
):
    tet = first_tet + wp.tid()
    materials[tet, 0] = shear[0]
    materials[tet, 1] = volume[0]


@wp.kernel
def _set_damping(value: wp.array(dtype=float), materials: wp.array2d(dtype=float)):
    materials[wp.tid(), 2] = value[0]


@wp.kernel
def _set_gravity(value: wp.array(dtype=float), gravity: wp.array(dtype=wp.vec3)):
    gravity[0] = wp.vec3(0.0, 0.0, -value[0])


@wp.kernel
def _set_cloth_bending(value: wp.array(dtype=float), materials: wp.array2d(dtype=float), first_edge: int):
    materials[first_edge + wp.tid(), 0] = value[0]


@wp.kernel
def _set_rigid_friction(value: wp.array(dtype=float), geom_friction: wp.array2d(dtype=wp.vec3), geom: int):
    geom_friction[0, geom] = wp.vec3(value[0], 0.005, 0.0001)


@wp.kernel
def _set_cart_force(value: wp.array(dtype=float), joint_f: wp.array(dtype=float)):
    joint_f[0] = value[0]


@wp.kernel
def _set_joint_pd(
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    target_ke: wp.array(dtype=float),
    target_kd: wp.array(dtype=float),
):
    target_ke[0] = stiffness[0]
    target_kd[0] = damping[0]


def _write_manifest(bundle: Path, demo: dict[str, object]) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["isaacLabDemo"] = demo
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")


def export_stiffness(output: Path) -> None:
    """Export three falling VBD cubes with independent Lamé controls for the middle cube.

    Args:
        output: Directory for the intermediate simulation bundle.
    """
    builder = newton.ModelBuilder()
    builder.default_particle_radius = 0.04
    builder.particle_max_velocity = 25.0
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0e-4, kf=1.0e3, mu=0.45))
    middle_tets = None
    for index, (x, stiffness, color) in enumerate(
        zip(
            (-0.8, 0.0, 0.8),
            (2.0e3, 2.0e4, 1.0e5),
            (wp.vec3(0.463, 0.725, 0.0), wp.vec3(0.333, 0.533, 0.039), wp.vec3(0.533, 0.8, 0.133)),
            strict=True,
        )
    ):
        first_tet = len(builder.tet_indices)
        builder.add_soft_grid(
            pos=wp.vec3(x - 0.24, -0.24, 1.55),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=3,
            dim_y=3,
            dim_z=3,
            cell_x=0.16,
            cell_y=0.16,
            cell_z=0.16,
            density=100.0,
            k_mu=stiffness,
            k_lambda=stiffness,
            k_damp=10.0,
            particle_radius=0.04,
            color=color,
        )
        if index == 1:
            middle_tets = (first_tet, len(builder.tet_indices))
    assert middle_tets is not None
    builder.color()
    model = builder.finalize(device="cpu")
    model.soft_contact_ke = 1.0e5
    model.soft_contact_kd = 1.0e-4
    model.soft_contact_kf = 1.0e3
    model.soft_contact_mu = 0.3
    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
        rigid_compliant_alm=True,
        rigid_body_particle_contact_buffer_size=128,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out = model.state(), model.state()
    collision = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision.contacts()
    shear = wp.array([2.0e4], dtype=float, device="cpu")
    volume = wp.array([2.0e4], dtype=float, device="cpu")
    damping = wp.array([10.0], dtype=float, device="cpu")
    gravity = wp.array([9.81], dtype=float, device="cpu")
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        wp.launch(
            _set_lame_parameters,
            dim=middle_tets[1] - middle_tets[0],
            inputs=[shear, volume, model.tet_materials, middle_tets[0]],
            device="cpu",
        )
        wp.launch(_set_damping, dim=len(builder.tet_indices), inputs=[damping, model.tet_materials], device="cpu")
        wp.launch(_set_gravity, dim=1, inputs=[gravity, model.gravity], device="cpu")
        state_in.clear_forces()
        collision.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, STIFFNESS_DT)
        wp.copy(state_in.particle_q, state_out.particle_q)
        wp.copy(state_in.particle_qd, state_out.particle_qd)
    export_graph(
        capture.graph,
        model=model,
        inputs={
            "particle_q": state_in.particle_q,
            "particle_qd": state_in.particle_qd,
            "shear": shear,
            "volume": volume,
            "damping": damping,
            "gravity": gravity,
        },
        outputs={"particle_q": state_in.particle_q, "particle_qd": state_in.particle_qd},
        output=output,
        timestep=STIFFNESS_DT,
        parameters=(
            Parameter("shear", 0, "Shear μ [Pa]", 1000.0, 200000.0, 1000.0),
            Parameter("volume", 0, "Volume λ [Pa]", 1000.0, 200000.0, 1000.0),
            Parameter("damping", 0, "Material damping", 0.0, 30.0, 0.5),
            Parameter("gravity", 0, "Gravity [m/s²]", 0.0, 20.0, 0.1),
        ),
        persistent=("shear", "volume", "damping", "gravity"),
    )
    _write_manifest(output, {"kind": "stiffness", "title": "Stiffness with VBD"})
    for _ in range(480):
        wp.capture_launch(capture.graph)
    if not np.isfinite(state_in.particle_q.numpy()).all():
        raise RuntimeError("VBD reference trajectory is not finite")


def export_cloth_bending(output: Path) -> None:
    """Export three VBD cloth sheets falling across pairs of rollers.

    Args:
        output: Directory for the intermediate simulation bundle.
    """
    builder = newton.ModelBuilder()
    builder.default_particle_radius = 0.018
    shape_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=100.0, mu=0.9)
    builder.add_ground_plane(cfg=shape_cfg)
    supports = []
    middle_edges = None
    roller_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)
    for index, (x_offset, stiffness, color) in enumerate(
        zip(
            (-1.55, 0.0, 1.55),
            (0.001, 1.0, 10.0),
            (wp.vec3(0.16, 0.48, 0.85), wp.vec3(0.57, 0.33, 0.85), wp.vec3(0.89, 0.48, 0.22)),
            strict=True,
        )
    ):
        for x in (-0.36, 0.36):
            position = (x + x_offset, 0.0, 0.55)
            builder.add_shape_cylinder(
                body=-1,
                xform=wp.transform(wp.vec3(*position), roller_rotation),
                radius=0.11,
                half_height=0.33,
                cfg=shape_cfg,
            )
            supports.append({"position": position, "radius": 0.11, "height": 0.66})
        first_edge = len(builder.edge_indices)
        builder.add_cloth_grid(
            pos=wp.vec3(x_offset - 0.6, -0.26, 0.8),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=10,
            dim_y=6,
            cell_x=0.12,
            cell_y=0.52 / 6,
            mass=0.002,
            tri_ke=3.0e3,
            tri_ka=3.0e3,
            tri_kd=1.0,
            edge_ke=stiffness,
            edge_kd=0.03,
            particle_radius=0.018,
            color=color,
        )
        if index == 1:
            middle_edges = (first_edge, len(builder.edge_indices))
    assert middle_edges is not None
    builder.color()
    model = builder.finalize(device="cpu")
    model.soft_contact_ke = 1.0e5
    model.soft_contact_kd = 100.0
    model.soft_contact_mu = 0.9
    solver = newton.solvers.SolverVBD(
        model,
        iterations=12,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
        rigid_body_particle_contact_buffer_size=512,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out = model.state(), model.state()
    collision = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision.contacts()
    bending = wp.array([1.0], dtype=float, device="cpu")
    gravity = wp.array([9.81], dtype=float, device="cpu")
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        wp.launch(
            _set_cloth_bending,
            dim=middle_edges[1] - middle_edges[0],
            inputs=[bending, model.edge_bending_properties, middle_edges[0]],
            device="cpu",
        )
        wp.launch(_set_gravity, dim=1, inputs=[gravity, model.gravity], device="cpu")
        state_in.clear_forces()
        collision.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, CLOTH_DT)
        wp.copy(state_in.particle_q, state_out.particle_q)
        wp.copy(state_in.particle_qd, state_out.particle_qd)

    class VisualModel:
        shape_type = wp.array([int(newton.GeoType.PLANE)], dtype=wp.int32, device="cpu")
        shape_body = wp.array([-1], dtype=wp.int32, device="cpu")
        shape_scale = wp.array([wp.vec3(8.0, 8.0, 0.0)], dtype=wp.vec3, device="cpu")
        shape_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")

        def __getattr__(self, name: str):
            return getattr(model, name)

    export_graph(
        capture.graph,
        model=VisualModel(),
        inputs={
            "particle_q": state_in.particle_q,
            "particle_qd": state_in.particle_qd,
            "bending": bending,
            "gravity": gravity,
        },
        outputs={"particle_q": state_in.particle_q, "particle_qd": state_in.particle_qd},
        output=output,
        timestep=CLOTH_DT,
        parameters=(
            Parameter("bending", 0, "Middle sheet bending [N·m]", 0.001, 10.0, 0.001),
            Parameter("gravity", 0, "Gravity [m/s²]", 0.0, 20.0, 0.1),
        ),
        persistent=("bending", "gravity"),
    )
    _write_manifest(
        output,
        {
            "kind": "cloth_bending",
            "title": "Cloth bending comparison with VBD",
            "supports": supports,
            "bendingScale": "log10",
            "cycleSteps": 360,
        },
    )
    for _ in range(360):
        wp.capture_launch(capture.graph)
    if not np.isfinite(state_in.particle_q.numpy()).all():
        raise RuntimeError("VBD cloth reference trajectory is not finite")


def export_rigid_friction(output: Path) -> None:
    """Export three boxes on an inclined MJWarp contact plane with live friction.

    Args:
        output: Directory for the intermediate simulation bundle.
    """
    builder = newton.ModelBuilder()
    slope = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.22)
    ramp = wp.transform(wp.vec3(0.0, 0.0, 0.7), slope)
    for y, friction in zip((-0.75, 0.0, 0.75), (0.05, 0.15, 0.8), strict=True):
        position = wp.transform_point(ramp, wp.vec3(-1.5, y, 0.25))
        body = builder.add_body(xform=wp.transform(position, slope))
        builder.add_shape_box(
            body=body,
            hx=0.22,
            hy=0.22,
            hz=0.22,
            cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=friction),
        )
    builder.add_shape_plane(xform=ramp, width=5.5, length=3.0, cfg=newton.ModelBuilder.ShapeConfig(mu=1.0e-4))
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(
        model,
        iterations=8,
        ls_iterations=16,
        use_mujoco_contacts=True,
        disable_sensors=True,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out = model.state(), model.state()
    middle_friction = wp.array([0.15], dtype=float, device="cpu")
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        wp.launch(
            _set_rigid_friction,
            dim=1,
            inputs=[middle_friction, solver.mjw_model.geom_friction, 2],
            device="cpu",
        )
        state_in.clear_forces()
        solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        wp.copy(state_in.joint_q, state_out.joint_q)
        wp.copy(state_in.joint_qd, state_out.joint_qd)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
    export_graph(
        capture.graph,
        model=model,
        inputs={
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "middle_friction": middle_friction,
        },
        outputs={
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
        },
        output=output,
        timestep=1.0 / 120.0,
        parameters=(Parameter("middle_friction", 0, "Middle box friction", 0.01, 1.0, 0.01),),
        persistent=("middle_friction",),
        colors=("#236bdb", "#8648ce", "#e34c31"),
    )
    _write_manifest(output, {"kind": "rigid_friction", "title": "Inclined friction with MJWarp", "cycleSteps": 240})
    for _ in range(240):
        wp.capture_launch(capture.graph)
    if not np.isfinite(state_in.body_q.numpy()).all():
        raise RuntimeError("MJWarp friction reference trajectory is not finite")


def export_joint_pd(output: Path) -> None:
    """Export a single revolute pendulum with live implicit-drive gains.

    Args:
        output: Directory for the intermediate simulation bundle.
    """
    builder = newton.ModelBuilder()
    pivot_height = 1.4
    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, pivot_height), wp.quat_identity()))
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, pivot_height), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        target_ke=30.0,
        target_kd=2.0,
        label="pendulum_hinge",
    )
    builder.add_articulation([joint])
    builder.add_shape_box(
        body=link,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.48), wp.quat_identity()),
        hx=0.07,
        hy=0.09,
        hz=0.48,
        cfg=newton.ModelBuilder.ShapeConfig(density=75.0),
        color=wp.vec3(0.28, 0.46, 0.88),
    )
    builder.joint_target_q[0] = 0.8
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, iterations=8, disable_sensors=True)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    stiffness = wp.array([30.0], dtype=float, device="cpu")
    damping = wp.array([2.0], dtype=float, device="cpu")
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        wp.launch(
            _set_joint_pd,
            dim=1,
            inputs=[stiffness, damping, model.joint_target_ke, model.joint_target_kd],
            device="cpu",
        )
        solver._update_joint_dof_properties()
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, 1.0 / 240.0)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        wp.copy(state_in.joint_q, state_out.joint_q)
        wp.copy(state_in.joint_qd, state_out.joint_qd)
    export_graph(
        capture.graph,
        model=model,
        inputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
            "target_q": control.joint_target_q,
            "stiffness": stiffness,
            "damping": damping,
        },
        outputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
        },
        output=output,
        timestep=1.0 / 240.0,
        parameters=(
            Parameter("target_q", 0, "Target angle [rad]", -1.2, 1.2, 0.05),
            Parameter("stiffness", 0, "Stiffness [N·m/rad]", 0.0, 120.0, 1.0),
            Parameter("damping", 0, "Damping [N·m·s/rad]", 0.0, 20.0, 0.1),
        ),
        persistent=("target_q", "stiffness", "damping"),
    )
    _write_manifest(output, {"kind": "joint_pd", "title": "Joint PD step response", "pivotHeight": pivot_height})
    for _ in range(480):
        wp.capture_launch(capture.graph)
    if not np.isfinite(state_in.joint_q.numpy()).all():
        raise RuntimeError("MJWarp joint PD reference trajectory is not finite")


def export_cartpole(output: Path, usd: Path, checkpoint: Path) -> None:
    """Export the Isaac Lab Cartpole asset and its trained MJWarp policy.

    Args:
        output: Directory for the intermediate simulation bundle.
        usd: Local copy of the Cartpole USD and its referenced ``Props`` directory.
        checkpoint: Published RSL-RL Cartpole checkpoint.
    """
    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

    task_cfg = CartpoleEnvCfg()
    mesh_path = usd.parent / "Props" / "instanceable_meshes.usd"
    if (
        hashlib.sha256(usd.read_bytes()).hexdigest() != CARTPOLE_USD_SHA256
        or hashlib.sha256(mesh_path.read_bytes()).hexdigest() != CARTPOLE_MESH_SHA256
    ):
        raise ValueError("Cartpole asset changed; review its joints and collision shapes before rebuilding")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CARTPOLE_CHECKPOINT_SHA256:
        raise ValueError("Cartpole checkpoint changed; review its task and actor before rebuilding")
    builder = newton.ModelBuilder()
    builder.add_usd(
        str(usd),
        xform=wp.transform(task_cfg.scene.cartpole.init_state.pos, wp.quat_identity()),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        load_visual_shapes=False,
        skip_mesh_approximation=True,
    )
    names = [label.rsplit("/", 1)[-1] for label in builder.joint_label]
    if names != [task_cfg.cart_dof_name, task_cfg.pole_dof_name]:
        raise ValueError(f"Cartpole joint order changed: {names}")
    builder.joint_q[builder.joint_q_start[1]] = 0.2
    builder.joint_damping[builder.joint_qd_start[0]] = task_cfg.scene.cartpole.actuators["cart_actuator"].damping
    builder.gravity = wp.vec3(*task_cfg.sim.gravity)
    model = builder.finalize(device="cpu")
    solver_cfg = task_cfg.sim.physics.newton_mjwarp.solver_cfg
    solver = newton.solvers.SolverMuJoCo(
        model,
        njmax=solver_cfg.njmax,
        nconmax=solver_cfg.nconmax,
        cone=solver_cfg.cone,
        impratio=solver_cfg.impratio,
        integrator=solver_cfg.integrator,
        use_mujoco_contacts=True,
    )
    state_in, state_out = model.state(), model.state()
    control = model.control()
    force = wp.array([0.0], dtype=float, device="cpu")
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        wp.launch(_set_cart_force, dim=1, inputs=[force, control.joint_f], device="cpu")
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, float(task_cfg.sim.dt))
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        wp.copy(state_in.joint_q, state_out.joint_q)
        wp.copy(state_in.joint_qd, state_out.joint_qd)

    class SkeletonModel:
        body_count = model.body_count
        particle_count = 0
        tri_count = 0
        shape_type = wp.array([int(newton.GeoType.PLANE)], dtype=wp.int32, device="cpu")
        shape_body = wp.array([-1], dtype=wp.int32, device="cpu")
        shape_scale = wp.array([wp.vec3(8.0, 8.0, 0.0)], dtype=wp.vec3, device="cpu")
        shape_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")

    export_graph(
        capture.graph,
        model=SkeletonModel(),
        inputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
            "force": force,
        },
        outputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
        },
        output=output,
        timestep=float(task_cfg.sim.dt),
        parameters=(Parameter("force", 0, "Cart force [N]", -500.0, 500.0, 1.0),),
        persistent=("force",),
    )
    shapes = []
    for shape_type, body, scale, transform in zip(
        builder.shape_type, builder.shape_body, builder.shape_scale, builder.shape_transform, strict=True
    ):
        if shape_type != newton.GeoType.BOX:
            raise ValueError("Expected only box collision shapes in Cartpole USD")
        shapes.append(
            {
                "body": body,
                "halfExtents": list(scale),
                "position": list(transform.p),
                "quaternion": list(transform.q),
            }
        )
    _write_manifest(
        output,
        {
            "kind": "cartpole",
            "title": "Cartpole policy with MJWarp",
            "task": "Isaac-Cartpole-Direct",
            "assetSha256": CARTPOLE_USD_SHA256,
            "shapes": shapes,
            "policy": _write_policy(checkpoint, output, (4, 32, 32, 1)),
            "decimation": task_cfg.decimation,
            "actionScale": task_cfg.action_scale,
            "perturbationLimit": 300.0,
            "maxCartPosition": task_cfg.max_cart_pos,
            "initialPoleAngle": 0.2,
        },
    )
    for _ in range(360):
        wp.capture_launch(capture.graph)
    if not np.isfinite(state_in.body_q.numpy()).all():
        raise RuntimeError("Cartpole reference trajectory is not finite")


def _configure_g1(
    builder: newton.ModelBuilder, description: dict
) -> tuple[list[str], list[float], list[int], list[int]]:
    """Match the WBC-AGILE policy's joint order, initial pose, and PD gains."""
    inputs = description["models"]["Velocity-G1-v0"]["inputs"]
    outputs = description["models"]["Velocity-G1-v0"]["outputs"]
    observation_names = next(item["element_names"][0] for item in inputs if item["name"] == "robot_joint_pos")
    action_names = next(item["element_names"][0] for item in outputs if item["name"] == "joint_pos")
    names = [label.rsplit("/", 1)[-1] for label in builder.joint_label[1:]]
    if len(names) != G1_DOF or set(names) != set(observation_names) or not set(action_names) <= set(names):
        raise ValueError("G1 asset joints do not match the WBC-AGILE policy")
    defaults = []
    builder.joint_q[2] = 0.8
    for index, name in enumerate(names, start=1):
        position = -0.1 if "hip_pitch" in name else 0.3 if "knee" in name else -0.2 if "ankle_pitch" in name else 0.0
        defaults.append(position)
        builder.joint_q[builder.joint_q_start[index]] = position
        if "hip_" in name:
            kp, kd = 100.0, 2.5
        elif "knee" in name:
            kp, kd = 200.0, 5.0
        elif "ankle_pitch" in name:
            kp, kd = 20.0, 0.2
        elif "ankle_roll" in name:
            kp, kd = 20.0, 0.1
        elif "waist" in name:
            kp, kd = 300.0, 5.0
        elif "shoulder_pitch" in name:
            kp, kd = 90.0, 2.0
        elif "shoulder_roll" in name:
            kp, kd = 60.0, 1.0
        elif "shoulder_yaw" in name:
            kp, kd = 20.0, 0.4
        elif "elbow" in name:
            kp, kd = 60.0, 1.0
        else:
            kp, kd = 4.0, 0.2
        dof = builder.joint_qd_start[index]
        builder.joint_target_ke[dof] = kp
        builder.joint_target_kd[dof] = kd
        builder.joint_armature[dof] = 0.02
    builder.joint_target_q[:] = builder.joint_q
    return (
        names,
        defaults,
        [names.index(name) for name in observation_names],
        [names.index(name) for name in action_names],
    )


def _write_policy(checkpoint: Path, output: Path, widths: tuple[int, ...]) -> dict[str, object]:
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    shapes = tuple(zip(widths[1:], widths[:-1], strict=True))
    chunks = []
    layers = []
    offset = 0
    for layer, (rows, columns) in enumerate(shapes):
        weight = weights[f"mlp.{layer * 2}.weight"].detach().numpy().astype("<f4", copy=False)
        bias = weights[f"mlp.{layer * 2}.bias"].detach().numpy().astype("<f4", copy=False)
        if weight.shape != (rows, columns) or bias.shape != (rows,):
            raise ValueError(f"Unexpected checkpoint layer {layer}: {weight.shape}, {bias.shape}")
        chunks.extend((weight.tobytes(), bias.tobytes()))
        layers.append({"rows": rows, "columns": columns, "offset": offset})
        offset += rows * (columns + 1)
    (output / "policy.bin").write_bytes(b"".join(chunks))
    return {"file": "policy.bin", "layers": layers, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def _write_agile_g1_policy(checkpoint: Path, output: Path) -> dict[str, object]:
    """Pack the published WBC-AGILE ONNX actor for the shared browser evaluator."""
    import onnx
    from onnx.numpy_helper import to_array

    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != G1_POLICY_SHA256:
        raise ValueError("G1 ONNX policy changed; review its observation and action contract")
    graph = onnx.load(checkpoint).graph
    arrays = {tensor.name: to_array(tensor) for tensor in graph.initializer}
    widths = (83, 256, 256, 128, 12)
    chunks, layers, offset = [], [], 0
    for index, (columns, rows) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        weight = np.asarray(arrays[f"_tensor_constant{5 + index * 2}"], dtype="<f4")
        bias = np.asarray(arrays[f"_tensor_constant{6 + index * 2}"], dtype="<f4")
        if weight.shape != (rows, columns) or bias.shape != (rows,):
            raise ValueError("Unexpected WBC-AGILE policy layer shape")
        chunks.extend((weight.tobytes(), bias.tobytes()))
        layers.append({"rows": rows, "columns": columns, "offset": offset})
        offset += rows * (columns + 1)
    (output / "policy.bin").write_bytes(b"".join(chunks))
    return {"file": "policy.bin", "layers": layers, "sha256": G1_POLICY_SHA256, "sourceLicense": "policy.LICENSE.txt"}


def _write_g1_visuals(source: Path, body_labels: list[str], joint_names: list[str], output: Path) -> dict[str, object]:
    """Pack simplified G1 link meshes into one lazy-loaded browser asset."""
    urdf_path = source / "g1_29dof_rev_1_0.urdf"
    if hashlib.sha256(urdf_path.read_bytes()).hexdigest() != G1_VISUAL_URDF_SHA256:
        raise ValueError("G1 visual URDF changed; review its link mapping before rebuilding")
    robot = ET.parse(urdf_path).getroot()
    joints = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    moving = {joint.get("name") for joint in joints.values() if joint.get("type") != "fixed"}
    if moving != set(joint_names):
        raise ValueError("G1 visual URDF joints do not match the simulation")
    body_indices = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(body_labels)}

    def origin_transform(element: ET.Element | None) -> np.ndarray:
        if element is None:
            return np.eye(4)
        xyz = [float(value) for value in element.get("xyz", "0 0 0").split()]
        rpy = [float(value) for value in element.get("rpy", "0 0 0").split()]
        transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
        transform[:3, 3] = xyz
        return transform

    def body_for_link(name: str) -> tuple[int, np.ndarray]:
        transform = np.eye(4)
        while name not in body_indices:
            joint = joints.get(name)
            if joint is None or joint.get("type") != "fixed":
                raise ValueError(f"No simulated body for G1 visual link {name}")
            transform = origin_transform(joint.find("origin")) @ transform
            name = joint.find("parent").get("link")
        return body_indices[name], transform

    data = bytearray()
    meshes = []
    for link in robot.findall("link"):
        visual = link.find("visual")
        mesh_element = visual.find("geometry/mesh") if visual is not None else None
        if mesh_element is None:
            continue
        body, transform = body_for_link(link.get("name"))
        transform = transform @ origin_transform(visual.find("origin"))
        filename = mesh_element.get("filename")
        mesh_path = source / filename
        mesh = trimesh.load_mesh(mesh_path, process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected a triangle mesh: {mesh_path}")
        scale = np.array([float(value) for value in mesh_element.get("scale", "1 1 1").split()])
        mesh.vertices *= scale
        mesh.apply_transform(transform)
        if len(mesh.faces) > 4000:
            mesh = mesh.simplify_quadric_decimation(face_count=4000, aggression=8)
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) >= 65536:
            raise ValueError(f"G1 visual mesh has too many vertices: {mesh_path}")
        vertices = np.asarray(mesh.vertices, dtype="<f4")
        faces = np.asarray(mesh.faces, dtype="<u2")
        if not np.isfinite(vertices).all():
            raise ValueError(f"Non-finite G1 visual mesh: {mesh_path}")
        vertex_offset = len(data)
        data.extend(vertices.tobytes())
        index_offset = len(data)
        data.extend(faces.tobytes())
        data.extend(bytes((-len(data)) % 4))
        name = link.get("name")
        material = (
            "accent"
            if name == "logo_link"
            else "shell"
            if name in {"pelvis", "pelvis_contour_link", "torso_link", "head_link"}
            else "structure"
        )
        meshes.append(
            {
                "body": body,
                "name": name,
                "material": material,
                "vertexOffset": vertex_offset,
                "vertexCount": len(vertices),
                "indexOffset": index_offset,
                "indexCount": faces.size,
            }
        )
    (output / "visuals.bin").write_bytes(data)
    return {
        "file": "visuals.bin",
        "byteLength": len(data),
        "meshes": meshes,
        "sourceRevision": G1_VISUAL_SOURCE_REVISION,
        "sourceLicense": "visuals.LICENSE.txt",
    }


def export_g1(output: Path, checkpoint: Path, description: Path, policy_license: Path, visual_source: Path) -> None:
    """Export the 29-joint Unitree G1 with the published WBC-AGILE velocity actor.

    Args:
        output: Directory for the intermediate simulation bundle.
        checkpoint: Published WBC-AGILE ``Velocity-G1-v0`` ONNX policy.
        description: Matching public WBC-AGILE policy YAML.
        policy_license: WBC-AGILE repository license file.
        visual_source: Unitree ROS G1 description and meshes at the pinned revision.
    """
    import yaml

    urdf_path = visual_source / "g1_29dof_rev_1_0.urdf"
    asset_sha256 = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if asset_sha256 != G1_VISUAL_URDF_SHA256:
        raise ValueError("G1 URDF changed; review its physics and visual mapping before rebuilding")
    policy_description = yaml.safe_load(description.read_text())
    model_description = policy_description["models"]["Velocity-G1-v0"]
    if model_description["parameters"]["sha256sum"] != G1_POLICY_SHA256:
        raise ValueError("G1 policy description does not match the pinned ONNX actor")

    # Keep the six links used for foot and upper-body contact in the reference setup.
    robot = ET.parse(urdf_path)
    for link in robot.getroot().findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        if link.get("name") not in G1_COLLISION_LINKS:
            for collider in link.findall("collision"):
                link.remove(collider)
    output.parent.mkdir(parents=True, exist_ok=True)
    physics_urdf = output.with_suffix(".physics.urdf")
    robot.write(physics_urdf)

    timestep = 0.001
    builder = newton.ModelBuilder()
    builder.default_body_armature = 0.02
    builder.default_joint_cfg.armature = 0.02
    builder.default_joint_cfg.target_ke = 20.0
    builder.default_joint_cfg.target_kd = 1.0
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e2
    builder.default_shape_cfg.mu = 1.0
    builder.add_urdf(str(physics_urdf), floating=True, collapse_fixed_joints=True, enable_self_collisions=False)
    physics_urdf.unlink()
    if len(builder.body_label) != 30 or len(builder.shape_type) != 12:
        raise ValueError("G1 physics model no longer has 30 bodies and 12 contact shapes")
    names, defaults, observation_indices, action_indices = _configure_g1(builder, policy_description)
    edges = [
        [parent, child] for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True) if parent >= 0
    ]
    torso_body = next(index for index, label in enumerate(builder.body_label) if label.endswith("/torso_link"))
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        njmax=400,
        nconmax=200,
        iterations=G1_BROWSER_SOLVER_ITERATIONS,
    )
    solver.mjw_model.opt.disableflags |= DisableBit.WARMSTART
    solver.mjw_model.opt.graph_conditional = False
    state_in, state_out = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    collision = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision.contacts()
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        state_in.clear_forces()
        collision.collide(state_in, contacts)
        solver._update_joint_dof_properties()
        solver.step(state_in, state_out, control, contacts, timestep)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        wp.copy(state_in.joint_q, state_out.joint_q)
        wp.copy(state_in.joint_qd, state_out.joint_qd)

    # The export metadata contains only the ground; visual meshes are packed separately.
    class SkeletonModel:
        body_count = model.body_count
        particle_count = 0
        tri_count = 0
        shape_type = wp.array([int(newton.GeoType.PLANE)], dtype=wp.int32, device="cpu")
        shape_body = wp.array([-1], dtype=wp.int32, device="cpu")
        shape_scale = wp.array([wp.vec3(8.0, 8.0, 0.0)], dtype=wp.vec3, device="cpu")
        shape_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")

    export_graph(
        capture.graph,
        model=SkeletonModel(),
        inputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
            "target_q": control.joint_target_q,
        },
        outputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
        },
        output=output,
        timestep=timestep,
    )
    policy = _write_agile_g1_policy(checkpoint, output)
    visuals = _write_g1_visuals(visual_source, builder.body_label, names, output)
    (output / "visuals.LICENSE.txt").write_bytes((visual_source.parent.parent / "LICENSE").read_bytes())
    (output / "policy.LICENSE.txt").write_bytes(policy_license.read_bytes())
    _write_manifest(
        output,
        {
            "kind": "g1",
            "title": "G1 WBC-AGILE velocity control",
            "task": "Velocity-G1-v0",
            "assetSha256": asset_sha256,
            "policy": policy,
            "visuals": visuals,
            "jointNames": names,
            "jointDefaults": defaults,
            "observationIndices": observation_indices,
            "actionIndices": action_indices,
            "policyType": "wbc_agile_g1",
            "edges": edges,
            "rootBody": int(builder.joint_child[0]),
            "torsoBody": torso_body,
            "decimation": 20,
            "actionScale": 0.25,
            "solverIterations": G1_BROWSER_SOLVER_ITERATIONS,
        },
    )
    for _ in range(20):
        wp.capture_launch(capture.graph)
    if not np.isfinite(state_in.body_q.numpy()).all():
        raise RuntimeError("G1 reference trajectory is not finite")


def _write_anymal_visuals(usd: Path, body_labels: list[str], output: Path) -> dict[str, object]:
    """Pack the visual meshes referenced by the ANYmal-D USD into one browser asset."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd))
    if stage is None:
        raise ValueError(f"Could not open ANYmal-D USD: {usd}")
    cache = UsdGeom.XformCache()
    bodies = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(body_labels)}
    data = bytearray()
    meshes = []
    for link in stage.GetPrimAtPath("/anymal").GetChildren():
        visuals = link.GetChild("visuals")
        if not visuals:
            continue
        link_name = link.GetName()
        body_name = link_name.replace("FOOT", "SHANK") if link_name.endswith("FOOT") else link_name
        if body_name not in bodies:
            raise ValueError(f"Missing simulated body for ANYmal-D link {link_name}")
        body_prim = stage.GetPrimAtPath(f"/anymal/{body_name}")
        body_world_inverse = cache.GetLocalToWorldTransform(body_prim).GetInverse()
        for prim in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies()):
            if prim.GetTypeName() != "Mesh":
                continue
            mesh = UsdGeom.Mesh(prim)
            counts = mesh.GetFaceVertexCountsAttr().Get()
            if any(count != 3 for count in counts):
                raise ValueError(f"Expected triangles in ANYmal-D visual {prim.GetPath()}")
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            local = np.asarray(cache.GetLocalToWorldTransform(prim) * body_world_inverse, dtype=np.float64)
            vertices = (vertices @ local[:3, :3] + local[3, :3]).astype("<f4")
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype="<u2")
            if len(vertices) >= 65536 or not np.isfinite(vertices).all():
                raise ValueError(f"Invalid ANYmal-D visual mesh: {prim.GetPath()}")
            binding = prim.GetRelationship("material:binding").GetTargets()
            material_name = binding[0].name.lower() if binding else ""
            material = (
                "accent"
                if "shell" in material_name
                else "structure"
                if any(part in material_name for part in ("drive", "shank", "foot", "hip"))
                else "shell"
            )
            vertex_offset = len(data)
            data.extend(vertices.tobytes())
            index_offset = len(data)
            data.extend(faces.tobytes())
            data.extend(bytes((-len(data)) % 4))
            meshes.append(
                {
                    "body": bodies[body_name],
                    "name": f"{link_name}/{prim.GetName()}",
                    "material": material,
                    "vertexOffset": vertex_offset,
                    "vertexCount": len(vertices),
                    "indexOffset": index_offset,
                    "indexCount": len(faces),
                }
            )
    if len(meshes) != 38:
        raise ValueError(f"Expected 38 ANYmal-D visual meshes; found {len(meshes)}")
    (output / "visuals.bin").write_bytes(data)
    return {"file": "visuals.bin", "byteLength": len(data), "meshes": meshes, "sourceLicense": "visuals.LICENSE.txt"}


def export_anymal(output: Path, usd: Path, checkpoint: Path) -> None:
    """Export the flat ANYmal-D policy with a browser PD actuator approximation.

    Args:
        output: Directory for the intermediate simulation bundle.
        usd: Local copy of the task's ANYmal-D USD and its referenced ``Props`` directory.
        checkpoint: Local copy of the published Newton MJWarp RSL-RL checkpoint.
    """
    from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg

    task_cfg = AnymalDFlatEnvCfg()
    terms = [name for name, term in vars(task_cfg.observations.policy).items() if getattr(term, "func", None)]
    expected_terms = [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    if terms != expected_terms:
        raise ValueError(f"ANYmal-D policy observation contract changed: {terms}")
    asset_sha256 = hashlib.sha256(usd.read_bytes()).hexdigest()
    mesh_path = usd.parent / "Props" / "instanceable_meshes.usd"
    if (
        asset_sha256 != ANYMAL_D_USD_SHA256
        or hashlib.sha256(mesh_path.read_bytes()).hexdigest() != ANYMAL_D_MESH_SHA256
        or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != ANYMAL_D_CHECKPOINT_SHA256
    ):
        raise ValueError("ANYmal-D asset or checkpoint changed; review its policy contract before rebuilding")
    solver_cfg = task_cfg.sim.physics.newton_mjwarp.solver_cfg
    if solver_cfg.use_mujoco_contacts:
        raise ValueError("ANYmal-D task collision mode changed; review browser collision capture")
    builder = newton.ModelBuilder()
    builder.default_body_armature = 0.01
    builder.default_joint_cfg.armature = 0.01
    shape_cfg = task_cfg.sim.physics.newton_mjwarp.default_shape_cfg
    builder.default_shape_cfg.ke = shape_cfg.ke
    builder.default_shape_cfg.kd = shape_cfg.kd
    builder.default_shape_cfg.mu = shape_cfg.mu
    builder.add_usd(
        str(usd),
        floating=True,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        load_visual_shapes=False,
        skip_mesh_approximation=True,
    )
    builder.gravity = wp.vec3(*task_cfg.sim.gravity)
    names = [label.rsplit("/", 1)[-1] for label in builder.joint_label[1:]]
    if len(names) != ANYMAL_D_DOF or len(set(names)) != ANYMAL_D_DOF:
        raise ValueError(f"Expected {ANYMAL_D_DOF} unique ANYmal-D joints; found {names}")
    builder.joint_q[2] = task_cfg.scene.robot.init_state.pos[2]
    defaults = []
    for index, name in enumerate(names, start=1):
        matching = [
            value for pattern, value in task_cfg.scene.robot.init_state.joint_pos.items() if re.fullmatch(pattern, name)
        ]
        if len(matching) > 1:
            raise ValueError(f"Ambiguous ANYmal-D default pose for {name}")
        position = float(matching[0]) if matching else 0.0
        builder.joint_q[builder.joint_q_start[index]] = position
        defaults.append(position)
        dof_index = builder.joint_qd_start[index]
        builder.joint_target_mode[dof_index] = newton.JointTargetMode.POSITION
        builder.joint_target_ke[dof_index] = ANYMAL_BROWSER_KP
        builder.joint_target_kd[dof_index] = ANYMAL_BROWSER_KD
    builder.joint_target_q[:] = builder.joint_q
    edges = [
        [parent, child] for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True) if parent >= 0
    ]
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(mu=task_cfg.scene.terrain.physics_material.dynamic_friction)
    )
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        njmax=solver_cfg.njmax,
        nconmax=80,
        iterations=8,
    )
    solver.mjw_model.opt.disableflags |= DisableBit.WARMSTART
    solver.mjw_model.opt.graph_conditional = False
    state_in, state_out = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    collision = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision.contacts()
    substeps = task_cfg.sim.physics.newton_mjwarp.num_substeps
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        for _ in range(substeps):
            state_in.clear_forces()
            collision.collide(state_in, contacts)
            solver._update_joint_dof_properties()
            solver.step(state_in, state_out, control, contacts, float(task_cfg.sim.dt) / substeps)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

    class SkeletonModel:
        body_count = model.body_count
        particle_count = 0
        tri_count = 0
        shape_type = wp.array([int(newton.GeoType.PLANE)], dtype=wp.int32, device="cpu")
        shape_body = wp.array([-1], dtype=wp.int32, device="cpu")
        shape_scale = wp.array([wp.vec3(8.0, 8.0, 0.0)], dtype=wp.vec3, device="cpu")
        shape_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")

    export_graph(
        capture.graph,
        model=SkeletonModel(),
        inputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
            "target_q": control.joint_target_q,
        },
        outputs={
            "body_q": state_in.body_q,
            "body_qd": state_in.body_qd,
            "joint_q": state_in.joint_q,
            "joint_qd": state_in.joint_qd,
        },
        output=output,
        timestep=float(task_cfg.sim.dt),
    )
    policy = _write_policy(checkpoint, output, (12 + 3 * ANYMAL_D_DOF, 128, 128, 128, ANYMAL_D_DOF))
    visuals = _write_anymal_visuals(usd, builder.body_label, output)
    (output / "visuals.LICENSE.txt").write_bytes(Path(__file__).with_name("anymal.LICENSE.txt").read_bytes())
    _write_manifest(
        output,
        {
            "kind": "anymal",
            "title": "ANYmal-D flat-ground velocity control",
            "task": "Isaac-Velocity-Flat-AnymalD",
            "assetSha256": asset_sha256,
            "policy": policy,
            "visuals": visuals,
            "jointNames": names,
            "jointDefaults": defaults,
            "edges": edges,
            "rootBody": int(builder.joint_child[0]),
            "torsoBody": int(builder.joint_child[0]),
            "decimation": task_cfg.decimation,
            "actionScale": task_cfg.actions.joint_pos.scale,
            "solverIterations": 8,
            "substeps": substeps,
            "actuator": {"type": "position_pd", "kp": ANYMAL_BROWSER_KP, "kd": ANYMAL_BROWSER_KD},
        },
    )
    for _ in range(20):
        wp.capture_launch(capture.graph)
    if not np.isfinite(state_in.body_q.numpy()).all():
        raise RuntimeError("ANYmal-D reference trajectory is not finite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "demo", choices=("stiffness", "cloth_bending", "rigid_friction", "joint_pd", "cartpole", "g1", "anymal")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--usd", type=Path, help="Local task USD asset; required for Cartpole and ANYmal-D")
    parser.add_argument("--checkpoint", type=Path, help="Local policy checkpoint (ONNX for G1, RSL-RL for others)")
    parser.add_argument("--policy-description", type=Path, help="WBC-AGILE G1 policy YAML")
    parser.add_argument("--policy-license", type=Path, help="WBC-AGILE repository license file")
    parser.add_argument("--visual-source", type=Path, help="Local pinned Unitree G1 description and meshes")
    parser.add_argument("--emxx", default="em++", help="Emscripten 5.0.3 compiler")
    args = parser.parse_args()
    wp.init()
    bundle = args.output / args.demo
    if args.demo == "stiffness":
        export_stiffness(bundle)
    elif args.demo == "cloth_bending":
        export_cloth_bending(bundle)
    elif args.demo == "rigid_friction":
        export_rigid_friction(bundle)
    elif args.demo == "joint_pd":
        export_joint_pd(bundle)
    elif args.demo == "cartpole":
        if args.usd is None or args.checkpoint is None:
            parser.error("Cartpole requires --usd and --checkpoint")
        export_cartpole(bundle, args.usd, args.checkpoint)
    elif args.demo == "g1":
        if any(
            value is None
            for value in (args.checkpoint, args.policy_description, args.policy_license, args.visual_source)
        ):
            parser.error("G1 requires --checkpoint, --policy-description, --policy-license, and --visual-source")
        export_g1(bundle, args.checkpoint, args.policy_description, args.policy_license, args.visual_source)
    else:
        if args.usd is None or args.checkpoint is None:
            parser.error("ANYmal-D requires --usd and --checkpoint")
        export_anymal(bundle, args.usd, args.checkpoint)
    subprocess.run(
        ["newton-web", "compile", str(bundle), "--output", str(args.output / f"{args.demo}-web"), "--emxx", args.emxx],
        check=True,
    )
    deployment = args.output / f"{args.demo}-web"
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["isaacLabDemo"] = json.loads((bundle / "manifest.json").read_text())["isaacLabDemo"]
    wasm_path = deployment / manifest["wasm"]
    if wasm_path.stat().st_size > 2_000_000:
        compressed_path = wasm_path.with_suffix(wasm_path.suffix + ".gz")
        with wasm_path.open("rb") as source, gzip.open(compressed_path, "wb", compresslevel=9) as destination:
            shutil.copyfileobj(source, destination)
        wasm_path.unlink()
        manifest["wasm"] = compressed_path.name
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    if args.demo in ("cartpole", "g1", "anymal"):
        (deployment / "policy.bin").write_bytes((bundle / "policy.bin").read_bytes())
    if args.demo in ("g1", "anymal"):
        (deployment / "visuals.bin").write_bytes((bundle / "visuals.bin").read_bytes())
        (deployment / "visuals.LICENSE.txt").write_bytes((bundle / "visuals.LICENSE.txt").read_bytes())
    if args.demo == "g1":
        (deployment / "policy.LICENSE.txt").write_bytes((bundle / "policy.LICENSE.txt").read_bytes())
    print(f"Built {args.demo}: {deployment}")


if __name__ == "__main__":
    main()
