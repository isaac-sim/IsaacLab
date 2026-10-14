# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture compact rigid and cable lessons from Isaac Lab's packaged examples."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import newton
import newton_web
import numpy as np
import warp as wp
from bundle import write_binary_files
from newton_web import Parameter, export_graph
from newton_web.compiler import compile_bundle

BUILD_VERSIONS = {"newton": "1.7.0.dev0", "warp-lang": "1.17.0", "mujoco-warp": "3.14.0"}
NEWTON_WEB_REVISION = "b0795fbe6b46e08a1fea2425699421415ca4cdf1"
FRAME_DT = 1.0 / 60.0


def validate_toolchain() -> dict[str, str]:
    """Use the same reviewed CPU runtime as the native MicroDuck export."""
    versions = {name: importlib.metadata.version(name) for name in BUILD_VERSIONS}
    if versions != BUILD_VERSIONS:
        raise ValueError(f"Expected {BUILD_VERSIONS}; found {versions}")
    compiler = Path(newton_web.__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(compiler), "rev-parse", "HEAD"], text=True).strip()
    if revision != NEWTON_WEB_REVISION:
        raise ValueError(f"Expected Newton Web {NEWTON_WEB_REVISION}; found {revision}")
    return {**versions, "newton-web": revision}


def package_bundle(bundle: Path, web: Path, emxx: str) -> dict:
    """Compile once and retain only browser runtime files in the deployment directory."""
    compile_bundle(bundle, web, emxx=emxx)
    path = web / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["build"].pop("modules", None)
    manifest["build"].pop("apicOperations", None)
    wasm = web / manifest["wasm"]
    files = write_binary_files(gzip.compress(wasm.read_bytes(), mtime=0), web, "simulation.wasm.gz")
    manifest["wasm"] = "simulation.wasm.gz"
    if len(files) > 1:
        manifest["wasmFiles"] = files
    wasm.unlink()
    path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    return manifest


@wp.kernel
def _domino_inputs(
    friction: wp.array(dtype=float),
    trigger: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
    dt: float,
):
    i = wp.tid()
    shape_mu[i] = friction[0]
    if i == 0 and trigger[0] > 0.0:
        # A short force at the top of the first slab starts the physical cascade.
        body_f[0] = wp.spatial_vector(2.0, 0.0, 0.0, 0.0, 0.32, 0.0)
        trigger[0] = wp.max(0.0, trigger[0] - dt)


@wp.kernel
def _cable_inputs(
    bending: wp.array(dtype=float),
    load: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    angular_dofs: wp.array(dtype=int),
    body_f: wp.array(dtype=wp.spatial_vector),
    tip: int,
):
    i = wp.tid()
    stiffness[angular_dofs[i]] = bending[0]
    if i == 0:
        body_f[tip] = wp.spatial_vector(0.0, 0.0, -load[0], 0.0, 0.0, 0.0)


def export_scene(name: str, output: Path) -> None:
    """Capture one complete 60 Hz frame, including native collision and substeps."""
    versions = validate_toolchain()
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.001
    if name == "dominoes":
        # Same slab dimensions, density, friction, restitution and solver as newton-dominoes.
        # Use one short row rather than the full 5,640-body logo.
        count = 24
        for index in range(count):
            x = (index - (count - 1) / 2) * 0.12
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.18), wp.quat_identity()))
            builder.add_shape_box(
                body,
                hx=0.016,
                hy=0.06,
                hz=0.18,
                cfg=newton.ModelBuilder.ShapeConfig(density=580.0, mu=1.0, restitution=0.15),
            )
        parameters = (Parameter("friction", 0, "Friction μ", 0.1, 1.5, 0.05),)
        values = {"friction": wp.array([1.0], dtype=float, device="cpu"), "trigger": wp.zeros(1, device="cpu")}
        persistent = ("friction",)
        title, source = "Domino cascade", "examples/newton_viewer_dominoes.py"
        options = {
            "action": {"label": "Topple", "binding": "trigger", "value": 0.18},
            "cameraTarget": [0, 0.25, 0],
            "cameraRadius": 3.8,
        }
        colors = ["#76b900" if i % 3 else "#2e6378" for i in range(count)]
        substeps = 8  # Compact CPU budget; validate cascade contacts against this native replay.
    else:
        # One cantilever uses the same native rod/VBD representation as the cable example.
        count = 20
        bodies, joints = builder.add_rod(
            rod=newton.Rod([wp.vec3(i * 0.9 / count - 0.45, 0.0, 1.0) for i in range(count + 1)], radius=0.01),
            cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.6),
            stretch_stiffness=5.0e5,
            bend_stiffness=1.0,
            bend_damping=0.05,
            body_frame_origin="com",
        )
        root = bodies[0]
        builder.body_mass[root] = builder.body_inv_mass[root] = 0.0
        builder.body_inertia[root] = builder.body_inv_inertia[root] = wp.mat33(0.0)
        # Native ROD stores [stretch, shear, bend, twist], rather than six D6 axes.
        dofs = [builder.joint_qd_start[joint] + builder.joint_dof_dim[joint][0] for joint in joints]
        if max(dofs) >= len(builder.joint_target_ke):
            raise ValueError("Native rod stiffness layout changed")
        angular_dofs = wp.array(dofs, dtype=int, device="cpu")
        parameters = (
            Parameter("bending", 0, "Bending stiffness [N·m/rad]", 0.001, 100.0, 0.001),
            Parameter("load", 0, "Downward tip load [N]", 0.0, 0.5, 0.01),
        )
        values = {
            "bending": wp.array([1.0], dtype=float, device="cpu"),
            "load": wp.array([0.05], dtype=float, device="cpu"),
        }
        persistent = tuple(values)
        title, source = "VBD cable bending", "examples/cables.py"
        options = {
            "logParameters": ["bending"],
            "cameraTarget": [0, 0.6, 0],
            "cameraRadius": 2.25,
            "cameraAzimuth": math.pi / 2,
            "cameraElevation": 0.1,
        }
        colors = ["#268a83"] * count
        substeps = 8
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0, restitution=0.15))
    builder.color()
    model = builder.finalize(device="cpu")
    iterations = 10 if name == "dominoes" else 20
    solver = (
        newton.solvers.SolverXPBD(model, iterations=iterations, enable_restitution=True)
        if name == "dominoes"
        else newton.solvers.SolverVBD(model, iterations=20, rigid_compliant_alm=False)
    )
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    control = model.control()
    dt = FRAME_DT / substeps
    with wp.ScopedCapture(device="cpu", apic=True) as capture:
        for _ in range(substeps):
            state_in.clear_forces()
            if name == "dominoes":
                wp.launch(
                    _domino_inputs,
                    model.shape_count,
                    [values["friction"], values["trigger"], model.shape_material_mu, state_in.body_f, dt],
                    device="cpu",
                )
            else:
                wp.launch(
                    _cable_inputs,
                    len(dofs),
                    [
                        values["bending"],
                        values["load"],
                        model.joint_target_ke,
                        angular_dofs,
                        state_in.body_f,
                        bodies[-1],
                    ],
                    device="cpu",
                )
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, dt)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
    bindings = {key: getattr(state_in, key) for key in ("body_q", "body_qd")}
    display = model
    if name == "cable_bending":
        # Newton Web's stock visual serializer does not handle capsules yet.
        # Publish their real geometry below and use the shared rigid viewer.
        display = SimpleNamespace(
            body_count=model.body_count,
            particle_count=0,
            tri_count=0,
            shape_type=wp.array([int(newton.GeoType.PLANE)], dtype=int, device="cpu"),
            shape_body=wp.array([-1], dtype=int, device="cpu"),
            shape_scale=wp.array([wp.vec3(4.0, 4.0, 0.0)], dtype=wp.vec3, device="cpu"),
            shape_transform=wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu"),
        )
    export_graph(
        capture.graph,
        model=display,
        inputs={**bindings, **values},
        outputs=bindings,
        output=output,
        timestep=FRAME_DT,
        parameters=parameters,
        persistent=persistent,
        colors=colors if name == "dominoes" else None,
    )
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if name == "cable_bending":
        manifest["visualization"]["shapes"] = [
            {
                "type": "capsule",
                "body": int(body),
                "radius": 0.01,
                "length": 0.9 / count,
                "transform": [0, 0, 0, 0, 0, 0, 1],
                "color": colors[index],
            }
            for index, body in enumerate(bodies)
        ]
        options["fixtures"] = [
            {"type": "box", "size": [0.08, 0.12, 0.16], "transform": [-0.45, 0, 0.98, 0, 0, 0, 1], "color": "#51626a"}
        ]
    manifest["isaacLabDemo"] = {
        "kind": name,
        "title": title,
        "viewer": "rigid",
        "controls": "parameters",
        "maxStepsPerFrame": 16,
        "source": source,
        "versions": versions,
        **options,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if name == "dominoes":
        values["trigger"].assign([0.18])
    reference = []
    for _ in range(240):
        wp.capture_launch(capture.graph)
        reference.append(state_in.body_q.numpy().copy())
    if not np.isfinite(reference).all():
        raise RuntimeError(f"{name}: non-finite captured trajectory")
    np.savez(output / "reference.npz", body_q=reference)
    print(f"Verified {name}: {model.body_count} bodies, {substeps} substeps × {iterations} iterations", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=("dominoes", "cable_bending"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emxx", default="em++")
    args = parser.parse_args()
    bundle = args.output / args.demo
    export_scene(args.demo, bundle)
    package_bundle(bundle, args.output / f"{args.demo}-web", args.emxx)


if __name__ == "__main__":
    main()
