# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Differentiable articulation geometry shared by reference motion, real states and imagined dynamics.

Native math owns quaternion conversion (xyzw). The smooth SO(3) exponential/logarithm
retains the derivatives required by the learned rollout.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_angle_axis, quat_from_matrix

from .math import matrix_to_quat, to_native
from .maxcoord import AnalyticIK


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Matrix to rotation vector with a finite zero-angle derivative for model fitting."""
    return axis_angle_from_quat(quat_from_matrix(R))


class ArticulationKinematics:
    """Differentiable hinge/slide articulation from an MJCF exported from the simulated asset.

    Fixed links and offset joint anchors are retained. Explicit joint/body names identify the
    first floating articulation, including names prefixed by Newton's USD exporter.
    """

    def __init__(self, mjcf_path: str, device: str, joint_names: list[str], tracked: list[str]):
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.device = device
        roots = [int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not roots:
            raise ValueError("Motion tracking requires a floating root in the kinematic model.")
        root = roots[0]
        bodies = [root]
        for b in range(root + 1, model.nbody):
            if int(model.body_parentid[b]) in bodies:
                bodies.append(b)
        local = {b: i for i, b in enumerate(bodies)}

        def match(name, candidates):
            if name in candidates:
                return candidates.index(name)
            found = [i for i, candidate in enumerate(candidates) if candidate == name or candidate.endswith("_" + name)]
            if len(found) != 1:
                raise ValueError(f"Expected one kinematic match for {name!r}, got {[candidates[i] for i in found]}.")
            return found[0]

        self.body_names = [model.body(b).name for b in bodies]
        self.tracked_idx = [match(name, self.body_names) for name in tracked]
        self.parent, self.joint_idx, self.joint_type = [], [], []
        axes, anchors, offsets, rotations, refs = [], [], [], [], []
        lower, upper = np.full(len(joint_names), -np.inf), np.full(len(joint_names), np.inf)
        used = set()
        for b in bodies:
            self.parent.append(-1 if b == root else local[int(model.body_parentid[b])])
            jid, count = int(model.body_jntadr[b]), int(model.body_jntnum[b])
            index, kind, ref = -1, -1, 0.0
            axis, anchor = np.zeros(3), np.zeros(3)
            if b != root and count:
                if count != 1 or int(model.jnt_type[jid]) not in (
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                ):
                    raise ValueError(f"Unsupported joint stack on {model.body(b).name}.")
                name = model.joint(jid).name
                found = [i for i, short in enumerate(joint_names) if name == short or name.endswith("_" + short)]
                if len(found) != 1 or found[0] in used:
                    raise ValueError(f"Kinematic joint {name!r} does not uniquely match the simulated joint layout.")
                index = found[0]
                used.add(index)
                kind = int(model.jnt_type[jid])
                ref = model.qpos0[model.jnt_qposadr[jid]]
                axis, anchor = model.jnt_axis[jid], model.jnt_pos[jid]
                if model.jnt_limited[jid]:
                    lower[index], upper[index] = model.jnt_range[jid]
            self.joint_idx.append(index)
            self.joint_type.append(kind)
            axes.append(axis.copy())
            anchors.append(anchor.copy())
            offsets.append(model.body_pos[b].copy())
            rotations.append(model.body_quat[b].copy())
            refs.append(ref)
        if len(used) != len(joint_names):
            raise ValueError(f"Missing kinematic joints: {set(range(len(joint_names))) - used}.")

        def tensor(values):
            return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

        self.axes, self.anchors, self.off_p = tensor(axes), tensor(anchors), tensor(offsets)
        self.off_R = matrix_from_quat(tensor(rotations)[:, [1, 2, 3, 0]])
        self.refs, self.q_lo, self.q_hi = tensor(refs), tensor(lower), tensor(upper)
        self.num_bodies = len(bodies)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        positions, rotations, velocities, angular = [p], [R], [v], [w]
        for i in range(1, self.num_bodies):
            parent, joint = self.parent[i], self.joint_idx[i]
            Rp, pp, vp, wp = rotations[parent], positions[parent], velocities[parent], angular[parent]
            origin = pp + (Rp @ self.off_p[i, :, None]).squeeze(-1)
            base_R = Rp @ self.off_R[i]
            Ri, pi, wi = base_R, origin, wp
            vi = vp + torch.cross(wp, origin - pp, dim=-1)
            if joint >= 0:
                axis = (base_R @ self.axes[i, :, None]).squeeze(-1)
                angle = q[..., joint] - self.refs[i]
                speed = qd[..., joint : joint + 1]
                if self.joint_type[i] == int(mujoco.mjtJoint.mjJNT_HINGE):
                    Ri = base_R @ matrix_from_quat(quat_from_angle_axis(angle, self.axes[i]))
                    pivot = origin + (base_R @ self.anchors[i, :, None]).squeeze(-1)
                    pi = pivot - (Ri @ self.anchors[i, :, None]).squeeze(-1)
                    wi = wp + axis * speed
                    vi = vp + torch.cross(wp, pivot - pp, dim=-1) + torch.cross(wi, pi - pivot, dim=-1)
                else:
                    pi = origin + axis * angle[..., None]
                    vi = vp + torch.cross(wp, pi - pp, dim=-1) + axis * speed
            positions.append(pi)
            rotations.append(Ri)
            velocities.append(vi)
            angular.append(wi)
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([positions[i] for i in idx], -2),
            torch.stack([rotations[i] for i in idx], -3),
            torch.stack([velocities[i] for i in idx], -2),
            torch.stack([angular[i] for i in idx], -2),
        )


class ForwardKinematics:
    """Adapt native articulation kinematics to the reference algorithm's callable interface."""

    def __init__(self, path: str, joint_names: list[str], body_names: list[str], device: str):
        self.kinematics = ArticulationKinematics(path, device, joint_names, body_names)
        self.path = path
        self.joint_names = joint_names
        self.body_names = body_names
        self.device = device

    def __call__(self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel):
        pos, rot, vel, ang_vel = self.kinematics.forward(
            root_pos, matrix_from_quat(to_native(root_quat)), root_lin_vel, root_ang_vel, joint_pos, joint_vel
        )
        return pos, matrix_to_quat(rot), vel, ang_vel

    def analytic_ik(self) -> AnalyticIK:
        """Build the reference maximal-coordinate joint reconstruction for this articulation."""
        model = mujoco.MjModel.from_xml_path(self.path)
        body_ids = torch.tensor([model.body(name).id for name in self.body_names])
        joint_q_adr = torch.tensor([model.jnt_qposadr[model.joint(name).id] for name in self.joint_names])
        return AnalyticIK(model, body_ids, joint_q_adr, torch.arange(len(self.body_names)), self.device, torch.float32)
