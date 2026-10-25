# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared NeRD representation for native recording, offline fitting and Newton inference.

PhysicsNeMo Robotics owns the joint codec, contact probes and network. Isaac Lab owns the
recorded control-period contract and the heading convention used by its locomotion tasks.
Import this module only when learned dynamics is selected; task discovery needs no PhysicsNeMo.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import fields

import newton
import numpy as np
import torch
import warp as wp
from physicsnemo_robotics.dynamics import (
    BodyHeadingFrame,
    BodyStateCodec,
    FixedContactFeatureAdapter,
    FixedRigidContactEncoder,
    JointStateCodec,
    expand_rigid_contact_shapes,
)
from physicsnemo_robotics.dynamics.codecs.joint import JointLayout

from isaaclab.utils.math import normalize, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul


class HeadingJointStateCodec(JointStateCodec):
    """Store world state, encode inputs and deltas in the current root's swing/twist heading frame."""

    def __init__(self, model=None, *, layout=None):
        super().__init__(model, layout=layout)
        if (
            not self.layout.root_is_free
            or self.layout.up_axis_index != 2
            or self.layout.free_root_translation_q_starts != (0,)
            or self.layout.quaternion_q_starts != (3,)
        ):
            raise ValueError("Heading dynamics requires one z-up free root at the start of the joint coordinates.")

    def encode_state(self, state):
        return super().encode_state(self._to_heading(state, state)[0])

    def state_to_delta(self, current, next_state):
        return super().state_to_delta(self._to_heading(current, current)[0], self._to_heading(next_state, current)[0])

    def delta_to_state(self, current, delta):
        local, heading, origin = self._to_heading(current, current)
        local = super().delta_to_state(local, delta)
        q, qd = local[..., : self.layout.dof_q], local[..., self.layout.dof_q :]
        return self.canonicalize_state(
            torch.cat(
                (
                    quat_apply(heading, q[..., :3]) + origin,
                    quat_mul(heading, q[..., 3:7]),
                    q[..., 7:],
                    quat_apply(heading, qd[..., :3]),
                    quat_apply(heading, qd[..., 3:6]),
                    qd[..., 6:],
                ),
                dim=-1,
            )
        )

    def _to_heading(self, state, anchor):
        twist = torch.cat((torch.zeros_like(anchor[..., 3:5]), anchor[..., 5:7]), dim=-1)
        identity = torch.zeros_like(twist)
        identity[..., 3] = 1.0
        heading = normalize(torch.where(twist.norm(dim=-1, keepdim=True) >= 1.0e-8, twist, identity))
        heading = torch.where(heading[..., 3:] < 0, -heading, heading)
        origin = torch.cat((anchor[..., :2], torch.zeros_like(anchor[..., 2:3])), dim=-1)
        q, qd = state[..., : self.layout.dof_q], state[..., self.layout.dof_q :]
        local = torch.cat(
            (
                quat_apply_inverse(heading, q[..., :3] - origin),
                quat_mul(quat_conjugate(heading), q[..., 3:7]),
                q[..., 7:],
                quat_apply_inverse(heading, qd[..., :3]),
                quat_apply_inverse(heading, qd[..., 3:6]),
                qd[..., 6:],
            ),
            dim=-1,
        )
        return local, heading, origin


class JointDynamics:
    """One representation used on stored tensors and, optionally, a live Newton model.

    Only uniform, joint-backed rigid worlds are supported. Contacts are explicit fixed sphere
    probes against a per-world plane; they are recomputed from each live predicted state.
    The descriptor is JSON data, so offline training does not initialize a simulation.
    """

    def __init__(
        self,
        descriptor=None,
        model=None,
        *,
        frame_dt=None,
        control=None,
        heading=False,
        contact_bodies=(),
        root_body="",
    ):
        if descriptor is None:
            if control not in ("joint_f", "joint_target_q"):
                raise ValueError("NeRD control must be joint_f or joint_target_q.")
            if (
                model.particle_count
                or model.body_count != model.joint_count
                or model.articulation_count != model.world_count
            ):
                raise ValueError("NeRD requires one complete joint-backed rigid articulation per world.")
            self.codec = HeadingJointStateCodec(model) if heading else JointStateCodec(model)
            layout = {}
            for field in fields(self.codec.layout):
                if field.init and field.name not in ("q_indices", "qd_indices", "semantic_signature"):
                    value = getattr(self.codec.layout, field.name)
                    layout[field.name] = value.tolist() if isinstance(value, torch.Tensor) else value
            layout["world_count"] = 1
            descriptor = {
                "version": 1,
                "frame_dt": frame_dt,
                "control": control,
                "heading": heading,
                "contact_bodies": list(contact_bodies),
                "root_body": root_body,
                "layout": layout,
                "model": model_signature(model),
            }
        else:
            layout = dict(descriptor["layout"])
            for name in ("continuous_q_mask", "root_translation_mask"):
                layout[name] = torch.tensor(layout[name], dtype=torch.bool)
            for name in ("quaternion_q_starts", "free_root_translation_q_starts"):
                layout[name] = tuple(layout[name])
            layout["world_count"] = 1 if model is None else model.world_count
            codec_type = HeadingJointStateCodec if descriptor["heading"] else JointStateCodec
            self.codec = codec_type(layout=JointLayout(**layout)) if model is None else codec_type(model)
        self.descriptor = descriptor
        self.contact_encoder = self.contact_adapter = None
        if model is not None and descriptor["contact_bodies"]:
            # The probe library compares local geometry across replicas. USD world-transform
            # roundoff is removed from a private view; live model arrays are never replaced.
            geometry = copy.copy(model)
            transforms = model.shape_transform.numpy().astype(np.float64)
            transforms[np.abs(transforms) < 1.0e-8] = 0.0
            geometry.shape_transform = wp.array(np.round(transforms, 6), dtype=wp.transform, device=model.device)
            body_ids = np.flatnonzero(model.body_world.numpy() == 0)
            body_names = [model.body_label[i].rsplit("/", 1)[-1] for i in body_ids]
            shape_ids = np.flatnonzero(model.shape_world.numpy() == 0)
            shape_body, shape_type = model.shape_body.numpy(), model.shape_type.numpy()
            feet = []
            for name in descriptor["contact_bodies"]:
                body = body_ids[body_names.index(name)]
                slots = [
                    i
                    for i, s in enumerate(shape_ids)
                    if shape_body[s] == body and shape_type[s] == newton.GeoType.SPHERE
                ]
                if len(slots) != 1:
                    raise ValueError(f"NeRD contact body {name!r} needs exactly one sphere.")
                feet.append(slots[0])
            planes = [
                i for i, s in enumerate(shape_ids) if shape_body[s] == -1 and shape_type[s] == newton.GeoType.PLANE
            ]
            if len(planes) != 1:
                raise ValueError("NeRD fixed contacts need exactly one plane in each world.")
            body_codec = BodyStateCodec(
                geometry, model_frame=BodyHeadingFrame(body=body_names.index(descriptor["root_body"]))
            )
            self.contact_encoder = FixedRigidContactEncoder(
                geometry,
                body_codec,
                points=expand_rigid_contact_shapes(geometry, body_codec, feet),
                environment_shape_slots=planes,
                environment_geometry="per_world",
                key="contact/probes",
            )
            self.contact_adapter = FixedContactFeatureAdapter(
                self.contact_encoder, ("point", "normal", "lever_arm", "gap", "relative_velocity", "contact_state")
            )

    @property
    def control_dim(self):
        return self.codec.layout.dof_q if self.descriptor["control"] == "joint_target_q" else self.codec.layout.dof_qd

    @property
    def contact_dim(self):
        return 14 * len(self.descriptor["contact_bodies"])

    @property
    def input_dim(self):
        return (
            self.codec.state_shape[0] + self.control_dim + (3 if self.descriptor["heading"] else 0) + self.contact_dim
        )

    def inputs(self, state, control, contacts):
        """Encode one frame or a batch of sequences without discarding autograd."""
        values = [self.codec.encode_state(state), control]
        if self.descriptor["heading"]:
            gravity = torch.zeros_like(state[..., :3])
            gravity[..., 2] = -1.0
            values.append(quat_apply_inverse(normalize(state[..., 3:7]), gravity))
        values.append(contacts)
        return torch.cat(values, dim=-1)

    def contacts(self, state):
        """Read fixed probes at the starting state, independently of the numerical solver's contacts."""
        if self.contact_encoder is None:
            return torch.empty(self.codec.batch_size, 0, device=state.joint_q.device.alias)
        return self.contact_adapter.encode_tokens(self.contact_encoder.encode_state(state))


def model_signature(model: newton.Model) -> str:
    """Fingerprint the supported rigid model without mutating it or including solver/viewer attributes.

    Replicated worlds must have the same physical parameters. Local transform roundoff from USD
    placement is tolerated at 1e-6; masses, gains, limits and friction remain exact. Free-root initial
    poses and per-world origins are state, and are excluded. Mesh geometry is included by content.
    """
    digest = hashlib.sha256()
    collision_shapes = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    groups = {
        "body": ("mass", "inertia", "com"),
        "joint": ("type", "dof_dim", "enabled", "X_p", "X_c"),
        "shape": (
            "type",
            "scale",
            "transform",
            "margin",
            "gap",
            "material_ke",
            "material_kd",
            "material_mu",
            "material_kf",
            "material_ka",
            "material_restitution",
            "material_mu_torsional",
            "material_mu_rolling",
        ),
    }
    for entity, names in groups.items():
        worlds = getattr(model, entity + "_world").numpy().copy()
        if entity == "shape":
            # Rendering can add world-positioned visual meshes. They do not change rigid dynamics.
            worlds[~collision_shapes] = -2
        counts = np.bincount(worlds[worlds >= 0], minlength=model.world_count)
        if np.any(counts != counts[0]):
            raise ValueError(f"NeRD needs identical {entity} counts in each world.")
        local = np.flatnonzero(worlds >= 0)
        ids = local[np.argsort(worlds[local], kind="stable")].reshape(model.world_count, -1)
        labels = getattr(model, entity + "_label")
        digest.update(json.dumps([labels[i].split("/env_0/")[-1] for i in ids[0]]).encode())
        for suffix in names:
            name = entity + "_" + suffix
            array = getattr(model, name).numpy().copy()
            if name == "joint_X_p":
                # World-parent transforms define reset origins, not local articulation geometry.
                array[model.joint_parent.numpy() == -1] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            if name == "shape_transform":
                # A horizontal plane is unchanged by translation in its own plane.
                planes = model.shape_type.numpy() == newton.GeoType.PLANE
                array[planes, :2] = 0.0
            if suffix in ("X_p", "X_c", "transform"):
                array = np.round(array.astype(np.float64), 6)
                array[np.abs(array) < 1.0e-8] = 0.0
            first = array[ids[0]]
            if not np.all(array[ids] == first):
                raise ValueError(f"NeRD cannot represent per-world variation in {name}.")
            digest.update(name.encode())
            digest.update(first.tobytes())
            digest.update(array[worlds == -1].tobytes())
    for name in (
        "joint_axis",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_armature",
        "joint_friction",
        "joint_target_ke",
        "joint_target_kd",
        "joint_damping",
        "joint_effort_limit",
        "joint_velocity_limit",
        "joint_limit_ke",
        "joint_limit_kd",
    ):
        values = getattr(model, name).numpy().reshape(model.world_count, -1)
        if not np.all(values == values[0]):
            raise ValueError(f"NeRD cannot represent per-world variation in {name}.")
        digest.update(name.encode())
        digest.update(values[0].tobytes())
    for name, values in (
        ("joint_parent", model.joint_parent.numpy()),
        ("joint_child", model.joint_child.numpy()),
        ("shape_body", model.shape_body.numpy()),
    ):
        entity = "shape" if name == "shape_body" else "joint"
        selected = getattr(model, entity + "_world").numpy() == 0
        if entity == "shape":
            selected &= collision_shapes
        indices = np.flatnonzero(selected)
        digest.update(name.encode())
        digest.update(values[indices].tobytes())
    for shape in np.flatnonzero((model.shape_world.numpy() == 0) & collision_shapes):
        source = model.shape_source[shape]
        if source is not None:
            if not isinstance(source, newton.Mesh):
                raise ValueError("NeRD supports primitive and mesh collision geometry only.")
            digest.update(np.asarray(source.vertices).tobytes())
            digest.update(np.asarray(source.indices).tobytes())
    gravity = model.gravity.numpy()
    if not np.all(gravity == gravity[0]):
        raise ValueError("NeRD cannot represent per-world gravity variation.")
    digest.update(np.asarray(gravity[0]).tobytes())
    return digest.hexdigest()
