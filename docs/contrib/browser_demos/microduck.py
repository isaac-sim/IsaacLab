# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the MicroDuck task's native BAM drives and walking policy for the browser."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import newton
import numpy as np
import onnx
import onnxruntime as ort
import torch
import warp as wp
from bundle import write_binary_files
from isaaclab_newton.physics import NewtonManager
from manipulation import _joint_coordinates, _pack_visuals
from newton_web import export_graph

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import get_current_stage
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_CFG, MICRODUCK_JOINT_NAMES

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
ISAACLAB_REVISION = "c47ef303990cfc117a9e0053a1a5e6864b46a513"
POLICY_URL = "https://huggingface.co/pollen-robotics/microduck-policies/resolve/v5/velstand.onnx"
POLICY_SHA256 = "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45"
BUILD_VERSIONS = {"newton": "1.7.0.dev0", "warp-lang": "1.17.0", "mujoco-warp": "3.14.0"}


@wp.kernel
def _apply_action(
    action: wp.array(dtype=float),
    slots: wp.array(dtype=int),
    defaults: wp.array(dtype=float),
    targets: wp.array2d(dtype=float),
):
    i = wp.tid()
    targets[0, slots[i]] = defaults[i] + action[i]


@wp.kernel
def _observe(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    dofs: wp.array(dtype=int),
    defaults: wp.array(dtype=float),
    action: wp.array(dtype=float),
    command: wp.array(dtype=float),
    last_velocity: wp.array(dtype=float),
    observation: wp.array(dtype=float),
    root: int,
):
    i = wp.tid()
    quat = wp.transform_get_rotation(pose[root])
    if i < 3:
        speed = velocity[root]
        angular = wp.quat_rotate_inv(quat, wp.vec3(speed[3], speed[4], speed[5]))
        observation[i] = angular[i]
    elif i < 6:
        gravity = wp.quat_rotate_inv(quat, wp.vec3(0.0, 0.0, -1.0))
        observation[i] = gravity[i - 3]
    elif i < 20:
        observation[i] = q[indices[i - 6]] - defaults[i - 6]
    elif i < 34:
        joint = i - 20
        observation[i] = last_velocity[joint]
        last_velocity[joint] = qd[dofs[joint]]
    elif i < 48:
        observation[i] = action[i - 34]
    else:
        observation[i] = command[i - 48]


@wp.kernel
def _push(force: wp.array(dtype=float), body_f: wp.array(dtype=wp.spatial_vector), root: int):
    body_f[root] = body_f[root] + wp.spatial_vector(force[0], force[1], 0.0, 0.0, 0.0, 0.0)


def _write_actor(path: Path, output: Path) -> dict:
    """Fold ONNX input normalization into its first dense layer; reuse the shared ELU evaluator."""
    graph = onnx.load(path).graph
    if [node.op_type for node in graph.node] != ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
        raise ValueError("MicroDuck actor graph changed; review the browser evaluator")
    tensors = {tensor.name: onnx.numpy_helper.to_array(tensor).copy() for tensor in graph.initializer}
    mean = tensors[graph.node[0].input[1]].reshape(-1)
    scale = tensors[graph.node[1].input[1]].reshape(-1)
    widths = (61, 512, 256, 128, 14)
    data, layers, offset = [], [], 0
    for layer, node in enumerate(node for node in graph.node if node.op_type == "Gemm"):
        weight, bias = tensors[node.input[1]], tensors[node.input[2]]
        rows, columns = widths[layer + 1], widths[layer]
        if weight.shape != (rows, columns) or bias.shape != (rows,):
            raise ValueError("MicroDuck actor widths changed")
        if layer == 0:
            weight = weight / scale
            bias = bias - weight @ mean
        data.extend((weight.astype("<f4").tobytes(), bias.astype("<f4").tobytes()))
        layers.append({"rows": rows, "columns": columns, "offset": offset})
        offset += rows * (columns + 1)
    files = write_binary_files(b"".join(data), output, "policy.bin")
    return {
        "file": files[0],
        "layers": layers,
        "source": POLICY_URL,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def export(output: Path, checkpoint: Path) -> None:
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != POLICY_SHA256:
        raise ValueError("MicroDuck policy changed; review its deployment contract before rebuilding")
    versions = {name: importlib.metadata.version(name) for name in BUILD_VERSIONS}
    if versions != BUILD_VERSIONS:
        raise ValueError(f"MicroDuck export requires {BUILD_VERSIONS}; found {versions}")
    torch.set_num_threads(1)
    cfg = parse_env_cfg(TASK, device="cpu", num_envs=1)
    cfg.play_mode()
    # Use the published whole-body collider variant so pushes and falls have real body contacts.
    cfg.scene.robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path=cfg.scene.robot.prim_path)
    cfg.seed = 42
    cfg.sim.physics.use_cuda_graph = False
    # Newton's lean support-map provider contains native pointers not relocated by this exporter.
    # MJWarp stores the same convex foot geometry in captured Warp arrays.
    cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
    cfg.sim.physics.solver_cfg.nconmax = 64
    cfg.sim.physics.solver_cfg.njmax = 256
    cfg.sim.physics.collision_cfg = None
    # One deterministic nominal robot; retain BAM's voltage, friction and command-delay kernels.
    cfg.events.encoder_bias.params["bias_range"] = (0.0, 0.0)
    cfg.events.imu_misalignment.params["max_angle_deg"] = 0.0
    for term in (cfg.observations.policy.base_ang_vel, cfg.observations.policy.projected_gravity):
        term.params["min_lag"] = term.params["max_lag"] = 0
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    for name in ("randomize_com", "randomize_head_com"):
        getattr(cfg.events, name).params["com_range"] = {}
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    for name in ("head_pose", "body_pose"):
        term = getattr(cfg.commands, name)
        term.ranges = tuple((0.0, 0.0) for _ in term.ranges)
        term.resampling_time_range = (1000.0, 1000.0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    actor = ort.InferenceSession(str(checkpoint), sess_options=options, providers=["CPUExecutionProvider"])
    with launch_simulation(cfg, {"visualizer": ["none"], "device": "cpu"}):
        env = ManagerBasedRLEnv(cfg)
        try:
            obs, _ = env.reset(seed=42)
            robot = env.scene["robot"]
            if env.action_manager.get_term("joint_pos")._joint_names != MICRODUCK_JOINT_NAMES:
                raise ValueError("MicroDuck action order changed")
            model = NewtonManager.get_model()
            coordinates, velocities = _joint_coordinates(model, MICRODUCK_JOINT_NAMES)
            slots = [robot.joint_names.index(name) for name in MICRODUCK_JOINT_NAMES]
            defaults_np = robot.data.default_joint_pos.warp.numpy()[0, slots]
            root = next(i for i, name in enumerate(model.body_label) if name.endswith("/trunk_base"))
            indices = wp.array(coordinates, dtype=int, device="cpu")
            dofs = wp.array(velocities, dtype=int, device="cpu")
            slot_indices = wp.array(slots, dtype=int, device="cpu")
            defaults = wp.array(defaults_np, dtype=float, device="cpu")
            action = wp.zeros(14, dtype=float, device="cpu")
            command = wp.zeros(13, dtype=float, device="cpu")
            previous_velocity = wp.zeros(14, dtype=float, device="cpu")
            observation = wp.zeros(61, dtype=float, device="cpu")
            force = wp.zeros(2, dtype=float, device="cpu")
            targets = robot.actuators.target_command.position.warp
            NewtonManager.register_state_force_callback(
                lambda state: wp.launch(_push, dim=1, inputs=[force, state.body_f, root], device="cpu")
            )
            state = NewtonManager.get_state_0()
            observe_args = [
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
                indices,
                dofs,
                defaults,
                action,
                command,
                previous_velocity,
                observation,
                root,
            ]
            reference = []
            for step in range(160):
                velocity_command = (0.0, 0.0, 0.0) if step < 40 else (0.35, 0.0, 0.0)
                env.command_manager.get_command("base_velocity")[:] = torch.tensor(velocity_command)
                command.assign([*velocity_command, *([0.0] * 10)])
                native_observation = obs["policy"].numpy()[0].copy()
                native_observation[48:51] = velocity_command
                wp.launch(_observe, dim=61, inputs=observe_args, device="cpu")
                np.testing.assert_allclose(observation.numpy(), native_observation, atol=2e-5)
                predicted = actor.run(None, {"obs": native_observation[None]})[0]
                action.assign(predicted[0])
                obs, _, done, _, _ = env.step(torch.from_numpy(predicted))
                if done.any():
                    raise RuntimeError("MicroDuck native walking validation terminated")
                reference.append(state.body_q.numpy().copy())
            print("Verified 160 native stand/walk steps and policy observations", flush=True)
            env.step(torch.zeros((1, 14)))
            env.reset(seed=42)
            action.zero_()
            command.zero_()
            previous_velocity.zero_()
            force.zero_()
            state = NewtonManager.get_state_0()
            observe_args[:4] = [state.body_q, state.body_qd, state.joint_q, state.joint_qd]
            wp.launch(_observe, dim=61, inputs=observe_args, device="cpu")
            with wp.ScopedCapture(device="cpu", apic=True) as capture:
                wp.launch(_apply_action, dim=14, inputs=[action, slot_indices, defaults, targets], device="cpu")
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                wp.launch(_observe, dim=61, inputs=observe_args, device="cpu")
            # Captured articulation views alias scene arrays; retain the largest owning region.
            regions = {}
            for key, region in capture.graph._apic_capture._regions.items():
                if region[0] not in regions or region[2] > regions[region[0]][1][2]:
                    regions[region[0]] = (key, region)
            capture.graph._apic_capture._regions = dict(regions.values())

            class DisplayModel:
                body_count = model.body_count
                particle_count = tri_count = 0
                shape_type = wp.array([int(newton.GeoType.PLANE)], dtype=int, device="cpu")
                shape_body = wp.array([-1], dtype=int, device="cpu")
                shape_scale = wp.array([wp.vec3(8.0, 8.0, 0.0)], dtype=wp.vec3, device="cpu")
                shape_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")

            bindings = {name: getattr(state, name) for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
            export_graph(
                capture.graph,
                model=DisplayModel(),
                inputs={**bindings, "action": action, "command": command, "force": force, "observation": observation},
                outputs={**bindings, "observation": observation},
                output=output,
                timestep=env.step_dt,
            )
            visuals = _pack_visuals(
                get_current_stage(), ["/World/envs/env_0/Robot"], model.body_label, output / "visuals.bin"
            )
            policy = _write_actor(checkpoint, output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["isaacLabDemo"] = {
                "kind": "microduck",
                "title": "MicroDuck walking",
                "task": TASK,
                "policyType": "microduck",
                "policy": policy,
                "visuals": visuals,
                "rootBody": root,
                "jointNames": MICRODUCK_JOINT_NAMES,
                "decimation": 1,
                "cameraRadius": 0.85,
                "cameraHeight": 0.13,
                "scene": "courtyard",
                "velocityScale": [0.4, 0.3, 1.0],
                "pushMaxForce": 2.0,
                "maxStepsPerFrame": 12,
                "physicsDt": cfg.sim.dt,
                "physicsDecimation": cfg.decimation,
                "actuator": "native BAM",
                "contacts": "MuJoCo Warp",
                "cameraAzimuth": 3.5,
                "source": {
                    "isaacLabRevision": ISAACLAB_REVISION,
                    "policy": POLICY_URL,
                    "asset": cfg.scene.robot.spawn.usd_path,
                    "assetSha256": hashlib.sha256(
                        Path(retrieve_file_path(cfg.scene.robot.spawn.usd_path)).read_bytes()
                    ).hexdigest(),
                    "versions": {
                        name: importlib.metadata.version(name)
                        for name in ("newton", "warp-lang", "mujoco-warp", "newton-web")
                    },
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            np.savez(output / "native_reference.npz", body_q=reference)
            # Review graph replay with the same ONNX actor before compilation.
            solver_time = NewtonManager._solver.mjw_data.time
            start_time = solver_time.numpy().copy()
            for step in range(160):
                command.assign([0.0 if step < 40 else 0.35, *([0.0] * 12)])
                observation_np = observation.numpy()
                observation_np[48:61] = command.numpy()
                observation.assign(observation_np)
                predicted = actor.run(None, {"obs": observation.numpy()[None]})[0]
                action.assign(predicted[0])
                wp.capture_launch(capture.graph)
                if step == 0:
                    np.testing.assert_allclose(solver_time.numpy() - start_time, env.step_dt, atol=2e-6)
                if not np.isfinite(state.body_q.numpy()).all() or state.body_q.numpy()[root, 2] < 0.06:
                    raise RuntimeError("MicroDuck captured walking validation failed")
            if state.body_q.numpy()[root, 0] < 0.02:
                raise RuntimeError("MicroDuck capture did not produce forward walking")
            print("Verified 160 captured stand/walk steps with native BAM", flush=True)
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emxx", default="em++")
    args = parser.parse_args()
    bundle = args.output / "microduck"
    export(bundle, args.checkpoint)
    subprocess.run(
        ["newton-web", "compile", str(bundle), "--output", str(args.output / "microduck-web"), "--emxx", args.emxx],
        check=True,
    )
    web = args.output / "microduck-web"
    manifest_path = web / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["build"].pop("modules", None)
    manifest["build"].pop("apicOperations", None)
    wasm = web / manifest["wasm"]
    files = write_binary_files(gzip.compress(wasm.read_bytes(), mtime=0), web, "simulation.wasm.gz")
    manifest["wasm"] = "simulation.wasm.gz"
    if len(files) > 1:
        manifest["wasmFiles"] = files
    wasm.unlink()
    (web / "policy.bin").write_bytes((bundle / "policy.bin").read_bytes())
    visual_files = write_binary_files(
        gzip.compress((bundle / "visuals.bin").read_bytes(), mtime=0), web, "visuals.bin.gz"
    )
    manifest["isaacLabDemo"]["visuals"]["file"] = "visuals.bin.gz"
    if len(visual_files) > 1:
        manifest["isaacLabDemo"]["visuals"]["files"] = visual_files
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
