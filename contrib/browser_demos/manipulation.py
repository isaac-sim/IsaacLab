# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture native Franka tasks with portable action and observation adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import newton
import numpy as np
import torch
import trimesh
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton_web import export_graph
from policy import write_policy
from rsl_rl.modules.normalization import EmpiricalNormalization

from pxr import Usd, UsdGeom, UsdShade

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import get_current_stage
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_tasks.utils import parse_env_cfg

# Each profile fixes the checkpoint contract, not the native scene or actuator settings.
MANIPULATION_TASKS = {
    "franka_reach": (
        "Isaac-Reach-Franka",
        (32, 64, 64, 7),
        "5465404261231eb08b979e8c02c756032866a6c098ff7fb25cde2bccb6e2b9c7",
    ),
    "franka_lift": (
        "Isaac-Lift-Franka",
        (1250, 512, 256, 128, 9),
        "2aca5190b3f3917ed96fcc49cd05c4e44da9b9e1c640a56db3040d75b82c8ce8",
    ),
    "franka_drawer": (
        "Isaac-Open-Drawer-Franka",
        (31, 256, 128, 64, 8),
        "fdb2b254e93bb3d47b7f78501f4ef8cbf849febe123a19e5e01abd8d87618d2e",
    ),
}


@wp.kernel
def _apply_action(
    kind: int,
    action: wp.array(dtype=float),
    q: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    defaults: wp.array(dtype=float),
    targets: wp.array2d(dtype=float),
):
    joint = wp.tid()
    if kind == 1:
        targets[0, joint] = q[indices[joint]] + 0.1 * action[joint]
    elif joint < 7:
        scale = wp.where(kind == 0, 0.5, 1.0)
        targets[0, joint] = defaults[joint] + scale * action[joint]
    elif kind == 2:
        targets[0, joint] = wp.where(action[7] < 0.0, 0.0, 0.04)


@wp.kernel
def _observe_frame(
    kind: int,
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    bodies: wp.array(dtype=wp.transform),
    indices: wp.array(dtype=int),
    dofs: wp.array(dtype=int),
    defaults: wp.array(dtype=float),
    root: int,
    hand: int,
    tips: wp.array(dtype=int),
    obj: int,
    drawer_q: int,
    drawer_dof: int,
    handle: int,
    points: wp.array(dtype=wp.vec3),
    right_force: wp.array2d(dtype=wp.vec3),
    left_force: wp.array2d(dtype=wp.vec3),
    right_friction: wp.array2d(dtype=wp.vec3),
    left_friction: wp.array2d(dtype=wp.vec3),
    command: wp.array(dtype=float),
    action: wp.array(dtype=float),
    frame: wp.array(dtype=float),
):
    feature = wp.tid()
    if kind == 0:
        if feature < 9:
            frame[feature] = q[indices[feature]] - defaults[feature]
        elif feature < 18:
            frame[feature] = qd[dofs[feature - 9]]
        elif feature < 25:
            frame[feature] = command[feature - 18]
        else:
            frame[feature] = action[feature - 25]
    elif kind == 2:
        if feature < 9:
            frame[feature] = q[indices[feature]] - defaults[feature]
        elif feature < 18:
            frame[feature] = qd[dofs[feature - 9]]
        elif feature == 18:
            frame[feature] = q[drawer_q]
        elif feature == 19:
            frame[feature] = qd[drawer_dof]
        elif feature < 23:
            ee = wp.transform_point(bodies[hand], wp.vec3(0.0, 0.0, 0.1034))
            target = wp.transform_point(bodies[handle], wp.vec3(0.305, 0.0, 0.01))
            frame[feature] = (target - ee)[feature - 20]
        else:
            frame[feature] = wp.clamp(action[feature - 23], -5.0, 5.0)
    else:
        inverse_root = wp.transform_inverse(bodies[root])
        if feature < 4:
            pose = inverse_root * bodies[obj]
            frame[feature] = wp.transform_get_rotation(pose)[feature]
        elif feature < 11:
            frame[feature] = command[feature - 4]
        elif feature < 20:
            frame[feature] = action[feature - 11]
        elif feature < 29:
            frame[feature] = q[indices[feature - 20]]
        elif feature < 38:
            frame[feature] = qd[dofs[feature - 29]]
        elif feature < 52:
            body = tips[(feature - 38) // 7]
            component = (feature - 38) % 7
            pose = inverse_root * bodies[body]
            value = float(0.0)
            if component < 3:
                value = wp.transform_get_translation(pose)[component]
            else:
                value = wp.transform_get_rotation(pose)[component - 3]
            frame[feature] = wp.clamp(value, -2.0, 2.0)
        elif feature < 58:
            force = right_force[0, 0] - right_friction[0, 0]
            if feature >= 55:
                force = left_force[0, 0] - left_friction[0, 0]
            force = wp.transform_vector(inverse_root, force)
            frame[feature] = wp.clamp(force[(feature - 52) % 3], -20.0, 20.0)
        else:
            point = wp.transform_point(bodies[obj], points[(feature - 58) // 3])
            point = wp.transform_point(inverse_root, point)
            frame[feature] = wp.clamp(point[(feature - 58) % 3], -2.0, 2.0)


@wp.kernel
def _record_observation(
    kind: int,
    frame: wp.array(dtype=float),
    history: wp.array2d(dtype=float),
    mean: wp.array(dtype=float),
    denominator: wp.array(dtype=float),
    observation: wp.array(dtype=float),
):
    feature = wp.tid()
    width = frame.shape[0]
    group_start = int(0)
    group_width = width
    if kind == 1:
        group_width = 4
        if feature >= 58:
            group_start = 58
            group_width = 192
        elif feature >= 52:
            group_start = 52
            group_width = 6
        elif feature >= 38:
            group_start = 38
            group_width = 14
        elif feature >= 29:
            group_start = 29
            group_width = 9
        elif feature >= 20:
            group_start = 20
            group_width = 9
        elif feature >= 11:
            group_start = 11
            group_width = 9
        elif feature >= 4:
            group_start = 4
            group_width = 7
    for time in range(history.shape[0] - 1):
        history[time, feature] = history[time + 1, feature]
    history[history.shape[0] - 1, feature] = frame[feature]
    for time in range(history.shape[0]):
        column = group_start * history.shape[0] + time * group_width + feature - group_start
        observation[column] = (history[time, feature] - mean[column]) / denominator[column]


def _joint_coordinates(model: newton.Model, names: list[str]) -> tuple[list[int], list[int]]:
    labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    coordinates, dofs = [], []
    q_start, dof_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    for name in names:
        if labels.count(name) != 1:
            raise ValueError(f"Expected one native joint named {name}")
        joint = labels.index(name)
        if q_start[joint + 1] - q_start[joint] != 1 or dof_start[joint + 1] - dof_start[joint] != 1:
            raise ValueError(f"Expected a scalar coordinate for {name}")
        coordinates.append(int(q_start[joint]))
        dofs.append(int(dof_start[joint]))
    return coordinates, dofs


def _body(model: newton.Model, name: str) -> int:
    matches = [index for index, label in enumerate(model.body_label) if label.rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one native body named {name}, found {matches}")
    return matches[0]


def _history_frames(raw: np.ndarray, kind: int) -> np.ndarray:
    if kind != 1:
        return raw.reshape(1, -1)
    widths = (4, 7, 9, 9, 9, 14, 6, 192)
    frames, offset = [], 0
    for width in widths:
        frames.append(raw[offset : offset + width * 5].reshape(5, width))
        offset += width * 5
    return np.concatenate(frames, axis=1)


def _actor_action(weights: dict[str, torch.Tensor], widths: tuple[int, ...], observation: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(observation.copy()).float()
    for layer in range(len(widths) - 1):
        value = torch.nn.functional.linear(value, weights[f"mlp.{layer * 2}.weight"], weights[f"mlp.{layer * 2}.bias"])
        if layer < len(widths) - 2:
            value = torch.nn.functional.elu(value)
    return value


def _check_contact_capacity() -> None:
    contacts = NewtonManager.get_contacts()
    if contacts is not None and contacts.rigid_contact_count.numpy()[0] >= contacts.rigid_contact_max:
        raise ValueError("Browser scene exhausted its rigid-contact capacity")


def _append_mesh(
    data: bytearray, meshes: list[dict], vertices: np.ndarray, faces: np.ndarray, body: int, name: str, color: tuple
) -> None:
    vertices, faces = vertices.astype("<f4"), faces.astype("<u2")
    if len(vertices) >= 65536 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid visual mesh: {name}")
    vertex_offset = len(data)
    data.extend(vertices.tobytes())
    index_offset = len(data)
    data.extend(faces.tobytes())
    data.extend(bytes((-len(data)) % 4))
    meshes.append(
        {
            "body": body,
            "name": name,
            "color": color,
            "vertexOffset": vertex_offset,
            "vertexCount": len(vertices),
            "indexOffset": index_offset,
            "indexCount": faces.size,
        }
    )


def _pack_cabinet_boxes(model: newton.Model, output: Path) -> dict[str, object]:
    """Draw procedural boxes around the native cabinet's individual colliders."""
    data, meshes = bytearray(), []
    types, bodies, scales, poses = [
        getattr(model, field).numpy() for field in ("shape_type", "shape_body", "shape_scale", "shape_transform")
    ]
    flags = model.shape_flags.numpy()
    for shape, body in enumerate(bodies):
        if (
            body < 0
            or "/Cabinet/" not in model.body_label[body]
            or not flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ):
            continue
        center = np.zeros(3)
        if types[shape] == int(newton.GeoType.BOX):
            extent = 2 * scales[shape]
        elif types[shape] == int(newton.GeoType.CONVEX_MESH):
            vertices = np.asarray(model.shape_source[shape].vertices) * scales[shape]
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            extent, center = upper - lower, (upper + lower) / 2
        else:
            raise ValueError("Cabinet visual adapter requires box or convex colliders")
        geometry = trimesh.creation.box(extents=extent)
        pose = poses[shape]
        rotation = trimesh.transformations.quaternion_matrix(pose[[6, 3, 4, 5]])[:3, :3]
        vertices = (geometry.vertices + center) @ rotation.T + pose[:3]
        color = (
            (0.12, 0.16, 0.19)
            if "handle" in model.body_label[body] or "nob" in model.body_label[body]
            else (0.72, 0.77, 0.8)
        )
        _append_mesh(data, meshes, vertices, geometry.faces, int(body), f"cabinet_{shape}", color)
    output.write_bytes(data)
    return {"file": output.name, "byteLength": len(data), "meshes": meshes}


def _pack_visuals(stage: Usd.Stage, roots: list[str], bodies: list[str], output: Path) -> dict[str, object]:
    cache = UsdGeom.XformCache()
    data, meshes = bytearray(), []
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            imageable = UsdGeom.Imageable(prim)
            if (
                not imageable
                or imageable.ComputeVisibility() == "invisible"
                or imageable.GetPurposeAttr().Get() == "guide"
            ):
                continue
            if prim.GetTypeName() == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                faces, start = [], 0
                for count in mesh.GetFaceVertexCountsAttr().Get():
                    faces.extend(
                        (indices[start], indices[start + j], indices[start + j + 1]) for j in range(1, count - 1)
                    )
                    start += count
                geometry = trimesh.Trimesh(vertices, faces, process=False)
                geometry.merge_vertices()
                if mesh.GetOrientationAttr().Get() == "leftHanded":
                    geometry.faces = geometry.faces[:, ::-1]
                if len(geometry.faces) > 1600:
                    geometry = geometry.simplify_quadric_decimation(face_count=1600)
            elif prim.GetTypeName() == "Cube":
                size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                geometry = trimesh.creation.box(extents=(size, size, size))
            else:
                continue
            path = str(prim.GetPath())
            parents = [
                (len(label), index)
                for index, label in enumerate(bodies)
                if path == label or path.startswith(label + "/")
            ]
            body = max(parents)[1] if parents else -1
            transform = cache.GetLocalToWorldTransform(prim)
            if body >= 0:
                transform *= cache.GetLocalToWorldTransform(stage.GetPrimAtPath(bodies[body])).GetInverse()
            matrix = np.asarray(transform, dtype=np.float64)
            vertices = (geometry.vertices @ matrix[:3, :3] + matrix[3, :3]).astype("<f4")
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            color = (0.72, 0.77, 0.8)
            if material:
                diffuse = material.GetInput("diffuseColor")
                if diffuse and diffuse.Get() is not None:
                    color = tuple(diffuse.Get())
                for shader_prim in Usd.PrimRange(material.GetPrim()):
                    if shader_prim.GetTypeName() == "Shader":
                        diffuse = UsdShade.Shader(shader_prim).GetInput("diffuseColor")
                        if diffuse and diffuse.Get() is not None:
                            color = tuple(diffuse.Get())
                            break
            _append_mesh(data, meshes, vertices, geometry.faces, body, prim.GetName(), color)
    output.write_bytes(data)
    return {"file": output.name, "byteLength": len(data), "meshes": meshes}


def export_manipulation(name: str, output: Path, checkpoint: Path) -> None:
    """Capture one native Newton task with deterministic browser playback.

    Args:
        name: Reviewed manipulation profile.
        output: Directory receiving the intermediate bundle.
        checkpoint: Published Newton RSL-RL checkpoint for this task.
    """
    task, widths, expected_sha = MANIPULATION_TASKS[name]
    torch.set_num_threads(1)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"{task} checkpoint changed; review its policy contract before rebuilding")
    kind = ("franka_reach", "franka_lift", "franka_drawer").index(name)
    overrides = ["physics=newton_mjwarp"]
    if kind == 1:
        overrides.append("presets=cube")
    cfg = parse_env_cfg(task, device="cpu", num_envs=1, overrides=overrides)
    cfg.play_mode()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.sim.physics.use_cuda_graph = False
    if kind == 1:
        # The native task reserves four million contacts for thousands of worlds.
        cfg.sim.physics.collision_cfg.rigid_contact_max = 512
    if kind == 0:
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    if kind != 2:
        command_cfg = cfg.commands.ee_pose if kind == 0 else cfg.commands.object_pose
        command_cfg.debug_vis = False
        command_cfg.resampling_time_range = (1000.0, 1000.0)
        position = (0.5, 0.0, 0.35) if kind == 0 else (0.55, 0.0, 0.7)
        command_cfg.ranges.pos_x, command_cfg.ranges.pos_y, command_cfg.ranges.pos_z = [(v, v) for v in position]
        command_cfg.ranges.roll = command_cfg.ranges.yaw = (0.0, 0.0)
    else:
        cfg.scene.cabinet_frame.debug_vis = False
    with launch_simulation(cfg, {"visualizer": ["none"], "device": "cpu"}):
        env = ManagerBasedRLEnv(cfg)
        try:
            obs, _ = env.reset(seed=42)
            robot = env.scene["robot"]
            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            native_actor = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
            native_groups = ("policy", "proprio", "perception") if kind == 1 else ("policy",)
            raw_obs = torch.cat([obs[group] for group in native_groups], dim=-1).numpy()[0]
            if raw_obs.shape != (widths[0],) or robot.joint_names != [f"panda_joint{i}" for i in range(1, 8)] + [
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]:
                raise ValueError(f"{task} observation or joint ordering changed")
            coordinates, dofs = _joint_coordinates(model, robot.joint_names)
            indices = wp.array(coordinates, dtype=int, device="cpu")
            dof_indices = wp.array(dofs, dtype=int, device="cpu")
            defaults = wp.array(robot.data.default_joint_pos.numpy()[0], dtype=float, device="cpu")
            action = wp.zeros(widths[-1], dtype=float, device="cpu")
            command_values = (
                env.command_manager.get_command("ee_pose" if kind == 0 else "object_pose").numpy()[0]
                if kind != 2
                else np.zeros(7)
            )
            command = wp.array(command_values, dtype=float, device="cpu")
            root = _body(model, "panda_link0")
            hand = _body(model, "panda_hand")
            tips = wp.array(
                [_body(model, "panda_rightfinger"), _body(model, "panda_leftfinger")], dtype=int, device="cpu"
            )
            obj = _body(model, "Object") if kind == 1 else root
            drawer_q, drawer_dof, handle = 0, 0, root
            if kind == 2:
                q_indices, v_indices = _joint_coordinates(model, ["drawer_top_joint"])
                drawer_q, drawer_dof = q_indices[0], v_indices[0]
                handle = _body(model, "drawer_handle_top")
            points = wp.zeros(64, dtype=wp.vec3, device="cpu")
            forces = [wp.zeros((1, 1), dtype=wp.vec3, device="cpu") for _ in range(4)]
            if kind == 1:
                tip_cfg = env.observation_manager.cfg.proprio.hand_tips_state_b.params["body_asset_cfg"]
                tips = wp.array([_body(model, robot.body_names[i]) for i in tip_cfg.body_ids], dtype=int, device="cpu")
                term = env.observation_manager.cfg.perception.object_point_cloud.func
                points = wp.array(term.points_local.numpy()[0], dtype=wp.vec3, device="cpu")
                views = [env.scene.sensors[f"panda_{side}finger_object_s"].contact_view for side in ("right", "left")]
                forces = [view.force_matrix for view in views] + [view.force_matrix_friction for view in views]
            history_values = _history_frames(raw_obs, kind)
            frame = wp.array(history_values[-1], dtype=float, device="cpu")
            history = wp.array(history_values, dtype=float, device="cpu")
            mean_values, denominator_values = np.zeros(widths[0]), np.ones(widths[0])
            if "obs_normalizer._mean" in native_actor:
                mean_values = native_actor["obs_normalizer._mean"].numpy()[0]
                denominator_values = (
                    native_actor["obs_normalizer._std"].numpy()[0] + EmpiricalNormalization(widths[0]).eps
                )
            mean = wp.array(mean_values, dtype=float, device="cpu")
            denominator = wp.array(denominator_values, dtype=float, device="cpu")
            observation = wp.array((raw_obs - mean_values) / denominator_values, dtype=float, device="cpu")
            observation_args = [
                kind,
                state.joint_q,
                state.joint_qd,
                state.body_q,
                indices,
                dof_indices,
                defaults,
                root,
                hand,
                tips,
                obj,
                drawer_q,
                drawer_dof,
                handle,
                points,
                *forces,
                command,
                action,
                frame,
            ]
            record_args = [kind, frame, history, mean, denominator, observation]
            # Use the native command buffer and retain its actuator pipeline in the capture.
            targets = robot.actuators.target_command.position.warp
            wp.launch(_observe_frame, dim=frame.size, inputs=observation_args, device="cpu")
            np.testing.assert_allclose(frame.numpy(), history_values[-1], atol=2.0e-5, rtol=2.0e-5)
            reference_q, reference_body = [], []
            # Check the portable observation/history adapter against native manager output
            # along a learned-policy trajectory, including grasp contact.
            for _ in range(180):
                actor_input = _actor_action(native_actor, widths, (raw_obs - mean_values) / denominator_values)
                action.assign(actor_input.numpy())
                obs, *_ = env.step(actor_input.reshape(1, -1))
                _check_contact_capacity()
                raw_obs = torch.cat([obs[group] for group in native_groups], dim=-1).numpy()[0]
                wp.launch(_observe_frame, dim=frame.size, inputs=observation_args, device="cpu")
                np.testing.assert_allclose(frame.numpy(), _history_frames(raw_obs, kind)[-1], atol=2.0e-4, rtol=2.0e-4)
                wp.launch(_record_observation, dim=frame.size, inputs=record_args, device="cpu")
                np.testing.assert_allclose(
                    observation.numpy(), (raw_obs - mean_values) / denominator_values, atol=2.0e-3, rtol=2.0e-4
                )
                reference_q.append(state.joint_q.numpy().copy())
                reference_body.append(state.body_q.numpy().copy())
            print(f"Verified {task} observations for 180 native policy steps", flush=True)
            # Materialize solver allocations before capture, then restore the authored reset.
            env.step(torch.zeros((1, widths[-1])))
            obs, _ = env.reset(seed=42)
            action.zero_()
            state = NewtonManager.get_state_0()
            observation_args[1:4] = [state.joint_q, state.joint_qd, state.body_q]
            raw_obs = torch.cat([obs[group] for group in native_groups], dim=-1).numpy()[0]
            history_values = _history_frames(raw_obs, kind)
            history.assign(history_values)
            frame.assign(history_values[-1])
            observation.assign((raw_obs - mean_values) / denominator_values)
            if kind != 2:
                command.assign(env.command_manager.get_command("ee_pose" if kind == 0 else "object_pose").numpy()[0])
            NewtonManager.forward()
            physics_calls = 1 if NewtonManager.handles_decimation() else cfg.decimation
            print(f"Capturing {physics_calls} native physics calls per policy step", flush=True)
            with wp.ScopedCapture(device="cpu", apic=True) as capture:
                wp.launch(
                    _apply_action, dim=9, inputs=[kind, action, state.joint_q, indices, defaults, targets], device="cpu"
                )
                for _ in range(physics_calls):
                    env.scene.write_data_to_sim()
                    NewtonManager.step()
                wp.launch(_observe_frame, dim=frame.size, inputs=observation_args, device="cpu")
                wp.launch(_record_observation, dim=frame.size, inputs=record_args, device="cpu")
            # Native articulation views can share a captured allocation with the full scene.
            # Keep the owning capacity rather than the shorter robot-only alias.
            regions = {}
            for key, region in capture.graph._apic_capture._regions.items():
                previous = regions.get(region[0])
                if previous is None or region[2] > previous[1][2]:
                    regions[region[0]] = (key, region)
            capture.graph._apic_capture._regions = dict(regions.values())

            class DisplayModel:
                body_count = model.body_count
                particle_count = 0
                tri_count = 0
                shape_type = wp.array([int(newton.GeoType.PLANE)], dtype=wp.int32, device="cpu")
                shape_body = wp.array([-1], dtype=wp.int32, device="cpu")
                shape_scale = wp.array([wp.vec3(8.0, 8.0, 0.0)], dtype=wp.vec3, device="cpu")
                shape_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")

            bindings = {key: getattr(state, key) for key in ("body_q", "body_qd", "joint_q", "joint_qd")}
            export_graph(
                capture.graph,
                model=DisplayModel(),
                inputs={**bindings, "action": action, "command": command, "observation": observation},
                outputs={**bindings, "observation": observation},
                output=output,
                timestep=env.step_dt,
                persistent=("command",),
            )
            np.savez(output / "native_reference.npz", joint_q=reference_q, body_q=reference_body)
            policy = write_policy(checkpoint, output, widths)
            stage = get_current_stage()
            robot_stage = Usd.Stage.Open(retrieve_file_path(cfg.scene.robot.spawn.usd_path))
            body_paths = [f"/__unbound/{i}" for i in range(model.body_count)]
            for prim in robot_stage.Traverse():
                if prim.GetName() in robot.body_names:
                    body_paths[_body(model, prim.GetName())] = str(prim.GetPath())
            visuals = _pack_visuals(
                robot_stage, [str(robot_stage.GetDefaultPrim().GetPath())], body_paths, output / "visuals.bin"
            )
            visuals["file"] = "../shared/franka_visuals.bin"
            extra_roots = [
                "/World/envs/env_0/" + item
                for item in (("Object", "table") if kind == 1 else ("Cabinet",) if kind == 2 else ("Table",))
            ]
            extras = (
                _pack_cabinet_boxes(model, output / "scene.bin")
                if kind == 2
                else _pack_visuals(stage, extra_roots, model.body_label, output / "scene.bin")
            )
            if kind == 1:
                for visual in extras["meshes"]:
                    visual["color"] = (0.95, 0.4, 0.06) if visual["body"] == obj else (0.55, 0.6, 0.63)
            title = {0: "Franka reach policy", 1: "Franka lift policy", 2: "Franka drawer-opening policy"}[kind]
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["isaacLabDemo"] = {
                "kind": name,
                "title": title,
                "task": task,
                "policy": policy,
                "visuals": [visuals, extras],
                "rootBody": root,
                "handBody": hand,
                "objectBody": obj if kind == 1 else None,
                "drawerCoordinate": drawer_q if kind == 2 else None,
                "decimation": 1,
                "cycleSteps": int(8 / env.step_dt),
                "cameraTarget": [0.4, 0.35, 0.0] if kind == 0 else [-0.4, 0.4, 0.0] if kind == 1 else [0.5, 0.5, 0.0],
                "cameraRadius": 2.0,
                "groundHeight": -1.05 if kind == 0 else 0.0,
                "commandRange": [[0.35, 0.65], [-0.2, 0.2], [0.15, 0.5]]
                if kind == 0
                else [[0.35, 0.65], [-0.2, 0.2], [0.55, 0.85]]
                if kind == 1
                else None,
                "sourceAsset": cfg.scene.robot.spawn.usd_path,
            }
            manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
            graph_q, graph_body = [], []
            for _ in range(180):
                actor_input = _actor_action(native_actor, widths, observation.numpy())
                action.assign(actor_input.numpy())
                wp.capture_launch(capture.graph)
                _check_contact_capacity()
                graph_q.append(state.joint_q.numpy().copy())
                graph_body.append(state.body_q.numpy().copy())
            np.savez(output / "graph_reference.npz", joint_q=graph_q, body_q=graph_body)
        finally:
            env.close()
