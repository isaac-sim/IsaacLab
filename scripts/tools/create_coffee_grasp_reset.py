# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author a Rizon/Sharpa grasp reset and optionally screen free-spoon retention.

Start with ``uv run python scripts/tools/create_coffee_grasp_reset.py --validate``.
The JSON contains named reset positions and separate actuator preload targets.
Edit those values, then pass the file with ``--input`` to screen a new candidate.
The isolated validator uses the exact USD collision meshes, a dynamic 25 g
spoon, gravity and MuJoCo CPU physics. It has no mounting, particles or renderer.
A passing stationary screen does not certify grasp acquisition or MPM scooping.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from isaaclab_tasks.contrib.rizon_sharpa_coffee.grasp_reset import (
    CoffeeGraspReset,
    default_grasp_reset,
    load_grasp_reset,
)


def validate_retention(candidate: CoffeeGraspReset, duration_s: float = 2.0) -> dict:
    """Screen geometry and gravity retention with the spoon free to fall."""
    import mujoco
    import newton
    import warp as wp
    from isaaclab_newton.physics import NewtonMJWarpManager
    from newton.solvers import SolverMuJoCo

    from pxr import Usd, UsdPhysics

    from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import build_coffee_assets
    from isaaclab_tasks.contrib.rizon_sharpa_coffee.physics import configure_grasp_contacts
    from isaaclab_tasks.contrib.rizon_sharpa_coffee.physics_cfg import make_grasp_solver_cfg
    from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import PALM_BODY_NAME, configured_asset_path, get_robot_cfg

    candidate.validate()
    if not math.isfinite(duration_s) or duration_s < 1.0:
        raise ValueError("A retention screen requires at least one finite second.")
    dt = 1.0 / 1600.0
    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(str(configured_asset_path()), enable_self_collisions=False, load_visual_shapes=False)
        actuators = get_robot_cfg(contact_grasp=True).actuators
        for joint_id, label in enumerate(builder.joint_label):
            name = str(label).rsplit("/", 1)[-1]
            if name not in candidate.joint_positions:
                continue
            q_id, dof_id = builder.joint_q_start[joint_id], builder.joint_qd_start[joint_id]
            q, target = candidate.joint_positions[name], candidate.joint_targets[name]
            lower, upper = builder.joint_limit_lower[dof_id], builder.joint_limit_upper[dof_id]
            if not lower <= q <= upper or not lower <= target <= upper:
                raise ValueError(f"Reset or target for {name} exceeds the USD joint limit.")
            actuator = next(
                cfg
                for cfg in actuators.values()
                if any(re.fullmatch(pattern, name) for pattern in cfg.joint_names_expr)
            )
            builder.joint_q[q_id] = q
            builder.joint_target_q[dof_id] = target
            builder.joint_target_ke[dof_id] = actuator.stiffness
            builder.joint_target_kd[dof_id] = actuator.damping
            builder.joint_effort_limit[dof_id] = actuator.joint_effort_limit
            builder.joint_velocity_limit[dof_id] = actuator.joint_velocity_limit
            builder.joint_target_mode[dof_id] = newton.JointTargetMode.POSITION_VELOCITY
        palm_id = next(
            index for index, label in enumerate(builder.body_label) if str(label).endswith("/" + PALM_BODY_NAME)
        )
        robot_model = builder.finalize(device="cpu")
        robot_state = robot_model.state()
        newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state)
        palm = _transform(wp, robot_state.body_q.numpy()[palm_id])
        offset = wp.transform(candidate.spoon_offset_pos, candidate.spoon_offset_rot)
        spoon_pose = wp.transform_multiply(palm, offset)
        stage = Usd.Stage.Open(build_coffee_assets()["spoon"])
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/Asset")).GetKinematicEnabledAttr().Set(False)
        imported = builder.add_usd(stage, xform=spoon_pose, floating=True, load_visual_shapes=False)
        spoon_id = imported["path_body_map"]["/Asset"]
        if not 0.0249 < builder.body_mass[spoon_id] < 0.0251:
            raise ValueError("The retention screen expects the authored dynamic 25 g teaspoon.")
        configure_grasp_contacts(builder, range(builder.shape_count))
        model = builder.finalize(device="cpu")
        state, other, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver_cfg = make_grasp_solver_cfg()
        solver_cfg.use_mujoco_cpu = True
        solver = NewtonMJWarpManager._create_solver(model, solver_cfg)
        # Populate native collision data without changing the authored geometry.
        solver.step(state, other, control, None, 1.0e-9)
        native_model, native_data = solver.mj_model, solver.mj_data
        finger_geoms = {
            index: match.group(1)
            for index in range(native_model.ngeom)
            if (match := re.search(r"right_(thumb|index|middle|ring|pinky)_", native_model.geom(index).name))
        }
        spoon_geoms = [
            index for index in range(native_model.ngeom) if native_model.geom(index).name.startswith("/Asset/")
        ]
        spoon_names = [native_model.geom(index).name for index in spoon_geoms]
        if not all(any(f"/{part}" in name for name in spoon_names) for part in ("BowlCollision", "HandleCollision")):
            raise ValueError("The spoon bowl and handle must remain separate collision meshes.")
        finger_pairs = [
            (a, b) for a in finger_geoms for b in finger_geoms if a < b and finger_geoms[a] != finger_geoms[b]
        ]
        tool_pairs = [(a, b) for a in finger_geoms for b in spoon_geoms]

        def clearance(pairs):
            return min(mujoco.mj_geomDistance(native_model, native_data, a, b, 0.1, None) for a, b in pairs)

        finger_clearance, tool_clearance = clearance(finger_pairs), clearance(tool_pairs)
        max_distance = max_angle = 0.0
        minimum_finger_clearance = finger_clearance
        minimum_tool_clearance = tool_clearance
        sampled = held_samples = 0
        finite = True
        for step in range(math.ceil(duration_s / dt)):
            solver.step(state, other, control, None, dt)
            state, other = other, state
            if step % 10:
                continue
            poses = state.body_q.numpy()
            relative = wp.transform_multiply(
                wp.transform_inverse(_transform(wp, poses[palm_id])), _transform(wp, poses[spoon_id])
            )
            distance = float(np.linalg.norm(np.asarray(relative[:3]) - candidate.spoon_offset_pos))
            cosine = abs(float(np.dot(np.asarray(relative[3:]), candidate.spoon_offset_rot)))
            angle = 2.0 * math.acos(min(cosine, 1.0))
            finite = bool(np.isfinite(poses).all())
            max_distance, max_angle = max(max_distance, distance), max(max_angle, angle)
            minimum_finger_clearance = min(minimum_finger_clearance, clearance(finger_pairs))
            minimum_tool_clearance = min(minimum_tool_clearance, clearance(tool_pairs))
            if step * dt >= 0.2:
                normals = []
                for contact_id, contact in enumerate(native_data.contact[: native_data.ncon]):
                    a, b = map(int, contact.geom)
                    finger = (
                        a
                        if a in finger_geoms and b in spoon_geoms
                        else b
                        if b in finger_geoms and a in spoon_geoms
                        else None
                    )
                    if finger is not None:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(native_model, native_data, contact_id, force)
                        if force[0] > 0.01:
                            normal = np.asarray(contact.frame[:3]) * (1.0 if finger == a else -1.0)
                            normals.append((finger_geoms[finger], normal))
                sampled += 1
                held_samples += any(
                    np.dot(thumb_normal, other_normal) < -0.5
                    for thumb_name, thumb_normal in normals
                    if thumb_name == "thumb"
                    for other_name, other_normal in normals
                    if other_name != "thumb"
                )
            if not finite or distance > 0.10:
                break
        contact_fraction = held_samples / max(sampled, 1)
        passed = (
            finite
            and (step + 1) * dt >= duration_s
            and finger_clearance >= -0.0001
            and tool_clearance >= -0.0001
            and minimum_finger_clearance >= -0.0001
            and minimum_tool_clearance >= -0.0005
            and max_distance <= 0.015
            and max_angle <= math.radians(15)
            and contact_fraction >= 0.9
        )
        return dict(
            passed=passed,
            backend="Newton SolverMuJoCo CPU",
            mounting=False,
            gravity_m_s2=9.81,
            spoon_mass_kg=0.025,
            timestep_s=dt,
            simulated_duration_s=(step + 1) * dt,
            minimum_initial_finger_clearance_m=float(finger_clearance),
            minimum_initial_tool_clearance_m=float(tool_clearance),
            minimum_dynamic_finger_clearance_m=float(minimum_finger_clearance),
            minimum_dynamic_tool_clearance_m=float(minimum_tool_clearance),
            max_relative_translation_m=max_distance,
            max_relative_rotation_deg=math.degrees(max_angle),
            opposing_finger_contact_fraction=contact_fraction,
            opposing_finger_contact_samples=sampled,
            scope="stationary gravity retention only; no acquisition, disturbance or MPM contact certification",
        )


def _transform(wp, value):
    return wp.transform(value[:3], value[3:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("logs/coffee_grasp/reset.json"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    candidate = load_grasp_reset(args.input) if args.input else default_grasp_reset()
    if args.validate:
        candidate.contact_validation = validate_retention(candidate, args.duration)
    candidate.save(args.output)
    print(f"Saved grasp candidate: {args.output.resolve()}")
    if args.validate:
        print(json.dumps(candidate.contact_validation, indent=2))
        if not candidate.contact_validation["passed"]:
            raise SystemExit("Candidate saved, but free-spoon retention FAILED; keep the mounted demo assumption.")


if __name__ == "__main__":
    main()
