# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the native Ant task, including its joint-wrench observations and effort drives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import torch
import warp as wp
from bundle import write_binary_files, write_policy
from isaaclab_newton.physics import NewtonManager
from manipulation import _joint_coordinates, _pack_visuals
from newton_web import export_graph
from scenes import package_bundle, validate_toolchain

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.sim import get_current_stage
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-Ant"
ISAACLAB_REVISION = "380ddac0c5b1fed6a50b4e439c24071be8b798fb"
CHECKPOINT_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Ant_newtonmjwarp_none_rsl_rl.pt"
CHECKPOINT_SHA256 = "19d5506f1082e339f1e2f8fb7d3846de74c6193feb9386a5a9254d1ff9860cf6"


@wp.kernel
def _apply_action(action: wp.array(dtype=float), targets: wp.array2d(dtype=float), scale: float):
    i = wp.tid()
    targets[0, i] = wp.clamp(action[i] * scale, -scale, scale)


@wp.kernel
def _push(force: wp.array(dtype=float), body_f: wp.array(dtype=wp.spatial_vector), root: int):
    body_f[root] = body_f[root] + wp.spatial_vector(force[0], force[1], 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _observe(
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    coordinates: wp.array(dtype=int),
    dofs: wp.array(dtype=int),
    limits: wp.array(dtype=wp.vec2),
    default_qd: wp.array(dtype=float),
    feet: wp.array(dtype=int),
    force: wp.array2d(dtype=wp.vec3),
    torque: wp.array2d(dtype=wp.vec3),
    action: wp.array(dtype=float),
    heading: wp.array(dtype=float),
    observation: wp.array(dtype=float),
    root: int,
):
    i = wp.tid()
    pos = wp.transform_get_translation(poses[root])
    quat = wp.transform_get_rotation(poses[root])
    velocity = velocities[root]
    yaw = wp.atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))
    if i == 0:
        observation[i] = pos[2]
    elif i < 4:
        linear = wp.quat_rotate_inv(quat, wp.vec3(velocity[0], velocity[1], velocity[2]))
        observation[i] = linear[i - 1]
    elif i < 7:
        angular = wp.quat_rotate_inv(quat, wp.vec3(velocity[3], velocity[4], velocity[5]))
        observation[i] = angular[i - 4]
    elif i == 7:
        observation[i] = yaw
    elif i == 8:
        observation[i] = wp.atan2(
            2.0 * (quat[3] * quat[0] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[0] * quat[0] + quat[1] * quat[1])
        )
    elif i == 9:
        angle = wp.atan2(1000.0 * wp.sin(heading[0]) - pos[1], 1000.0 * wp.cos(heading[0]) - pos[0]) - yaw
        observation[i] = wp.atan2(wp.sin(angle), wp.cos(angle))
    elif i == 10:
        observation[i] = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))[2]
    elif i == 11:
        direction = wp.normalize(
            wp.vec3(1000.0 * wp.cos(heading[0]) - pos[0], 1000.0 * wp.sin(heading[0]) - pos[1], 0.0)
        )
        observation[i] = wp.dot(wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0)), direction)
    elif i < 20:
        index = i - 12
        bounds = limits[index]
        observation[i] = 2.0 * (q[coordinates[index]] - bounds[0]) / (bounds[1] - bounds[0]) - 1.0
    elif i < 28:
        index = i - 20
        observation[i] = 0.2 * (qd[dofs[index]] - default_qd[index])
    elif i < 52:
        index = i - 28
        component = index % 6
        foot = feet[index // 6]
        if component < 3:
            observation[i] = 0.1 * force[0, foot][component]
        else:
            observation[i] = 0.1 * torque[0, foot][component - 3]
    else:
        observation[i] = action[i - 52]


def export(output: Path, checkpoint: Path) -> None:
    versions = validate_toolchain()
    revision = subprocess.check_output(["git", "-C", str(ISAACLAB_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if revision != ISAACLAB_REVISION:
        raise ValueError(f"Expected Isaac Lab {ISAACLAB_REVISION}; review task changes before exporting {revision}")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Ant checkpoint changed; review its observation/action contract")
    torch.set_num_threads(1)
    cfg = parse_env_cfg(TASK, device="cpu", num_envs=1)
    cfg.play_mode()
    cfg.sim.physics.use_cuda_graph = False
    cfg.sim.physics.collision_cfg = None
    cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    actor = torch.nn.Sequential(
        torch.nn.Linear(60, 400),
        torch.nn.ELU(),
        torch.nn.Linear(400, 200),
        torch.nn.ELU(),
        torch.nn.Linear(200, 100),
        torch.nn.ELU(),
        torch.nn.Linear(100, 8),
    )
    actor.load_state_dict({k.removeprefix("mlp."): v for k, v in weights.items() if k.startswith("mlp.")})
    with launch_simulation(cfg, {"visualizer": ["none"], "device": "cpu"}):
        env = ManagerBasedRLEnv(cfg)
        try:
            obs, _ = env.reset(seed=42)
            robot, sensor = env.scene["robot"], env.scene["joint_wrench"]
            model, state = NewtonManager.get_model(), NewtonManager.get_state_0()
            root = model.body_label.index("/World/envs/env_0/Robot/torso")
            coordinates, dofs = _joint_coordinates(model, robot.joint_names)
            feet = env.observation_manager.cfg.policy.feet_body_forces.params["sensor_cfg"].body_ids
            action, observation = wp.zeros(8, device="cpu"), wp.zeros(60, device="cpu")
            heading, force = wp.zeros(1, device="cpu"), wp.zeros(2, device="cpu")
            targets = robot.actuators.target_command.effort.warp
            scale = float(cfg.actions.joint_effort.scale)
            observe_args = [
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
                wp.array(coordinates, dtype=int, device="cpu"),
                wp.array(dofs, dtype=int, device="cpu"),
                wp.array(robot.data.soft_joint_pos_limits.torch.numpy()[0], dtype=wp.vec2, device="cpu"),
                wp.array(robot.data.default_joint_vel.torch.numpy()[0], dtype=float, device="cpu"),
                wp.array(feet, dtype=int, device="cpu"),
                sensor.data.force.warp,
                sensor.data.torque.warp,
                action,
                heading,
                observation,
                root,
            ]
            with torch.inference_mode():
                for _ in range(180):
                    wp.launch(_observe, 60, observe_args, device="cpu")
                    np.testing.assert_allclose(observation.numpy(), obs["policy"].numpy()[0], atol=2e-5, rtol=1e-5)
                    predicted = actor(obs["policy"])
                    action.assign(predicted.numpy()[0])
                    obs, _, done, _, _ = env.step(predicted)
                    if done.any():
                        raise RuntimeError("Native Ant policy terminated during validation")
            print("Verified 180 native Ant steps and all 60 observations, including foot wrenches", flush=True)
            env.reset(seed=42)
            action.zero_()
            sensor._update_buffers_impl(wp.ones(1, dtype=bool, device="cpu"))
            wp.launch(_observe, 60, observe_args, device="cpu")
            mask = wp.ones(1, dtype=bool, device="cpu")
            NewtonManager.register_state_force_callback(
                lambda s: wp.launch(_push, 1, [force, s.body_f, root], device="cpu")
            )
            with wp.ScopedCapture(device="cpu", apic=True) as capture:
                wp.launch(_apply_action, 8, [action, targets, scale], device="cpu")
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                sensor._update_buffers_impl(mask)
                wp.launch(_observe, 60, observe_args, device="cpu")
            regions = {}
            for key, region in capture.graph._apic_capture._regions.items():
                if region[0] not in regions or region[2] > regions[region[0]][1][2]:
                    regions[region[0]] = (key, region)
            capture.graph._apic_capture._regions = dict(regions.values())
            display = SimpleNamespace(
                body_count=model.body_count,
                particle_count=0,
                tri_count=0,
                shape_type=wp.array([int(newton.GeoType.PLANE)], dtype=int, device="cpu"),
                shape_body=wp.array([-1], dtype=int, device="cpu"),
                shape_scale=wp.array([wp.vec3(8.0, 8.0, 0.0)], dtype=wp.vec3, device="cpu"),
                shape_transform=wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu"),
            )
            bindings = {name: getattr(state, name) for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
            export_graph(
                capture.graph,
                model=display,
                inputs={**bindings, "action": action, "observation": observation, "heading": heading, "force": force},
                outputs={**bindings, "observation": observation},
                output=output,
                timestep=env.step_dt,
                persistent=("heading",),
            )
            visuals = _pack_visuals(
                get_current_stage(), ["/World/envs/env_0/Robot"], model.body_label, output / "visuals.bin"
            )
            if not visuals["meshes"]:
                raise ValueError("Ant has no browser visual geometry")
            # A restrained display palette distinguishes the links; collision geometry is unchanged.
            for mesh in visuals["meshes"]:
                mesh["color"] = (
                    (0.035, 0.24, 0.28)
                    if mesh["body"] == root
                    else ((0.12, 0.14, 0.17) if "ankle" in mesh["name"] else (0.3, 0.46, 0.5))
                )
            policy = write_policy(checkpoint, output, (60, 400, 200, 100, 8))
            path = output / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["isaacLabDemo"] = {
                "kind": "ant",
                "title": "Ant walking",
                "task": TASK,
                "policyType": "captured",
                "controls": "heading",
                "policy": policy,
                "visuals": visuals,
                "rootBody": root,
                "jointNames": robot.joint_names,
                "decimation": 1,
                "pushMaxForce": 100.0,
                "cycleSteps": 960,
                "terminationHeight": 0.31,
                "toneMappingExposure": 1.1,
                "cameraRadius": 3.4,
                "cameraHeight": 0.4,
                "cameraAzimuth": 3.5,
                "maxStepsPerFrame": 24,
                "source": {
                    "taskRevision": revision,
                    "checkpoint": CHECKPOINT_URL,
                    "checkpointSha256": CHECKPOINT_SHA256,
                    "asset": cfg.scene.robot.spawn.usd_path,
                    "assetSha256": hashlib.sha256(
                        Path(retrieve_file_path(cfg.scene.robot.spawn.usd_path)).read_bytes()
                    ).hexdigest(),
                    "versions": versions,
                },
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            reference = []
            with torch.inference_mode():
                for _ in range(240):
                    action.assign(actor(torch.from_numpy(observation.numpy()[None])).numpy()[0])
                    wp.capture_launch(capture.graph)
                    reference.append(state.body_q.numpy().copy())
            if not np.isfinite(reference).all() or reference[-1][root, 2] < 0.31 or reference[-1][root, 0] < 1:
                raise RuntimeError("Captured Ant did not retain native walking behavior")
            np.savez(output / "reference.npz", body_q=reference)
            print("Verified 240 captured Ant policy steps with native wrench sensor and drives", flush=True)
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emxx", default="em++")
    args = parser.parse_args()
    bundle, web = args.output / "ant", args.output / "ant-web"
    export(bundle, args.checkpoint)
    manifest = package_bundle(bundle, web, args.emxx)
    (web / "policy.bin").write_bytes((bundle / "policy.bin").read_bytes())
    files = write_binary_files(gzip.compress((bundle / "visuals.bin").read_bytes(), mtime=0), web, "visuals.bin.gz")
    manifest["isaacLabDemo"]["visuals"].update(file="visuals.bin.gz", files=files)
    (web / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
