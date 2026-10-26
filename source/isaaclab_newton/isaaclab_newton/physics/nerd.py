# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared NeRD representation for native recording, offline fitting and Newton inference.

PhysicsNeMo Robotics owns state conversion, contact encoders and temporal inference. Isaac Lab
owns the recorded physics-step contract, physical identity policy and task feature selection.
Import this module only when learned dynamics is selected; task discovery needs no PhysicsNeMo.
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import nullcontext

import newton
import numpy as np
import torch
import warp as wp
from physicsnemo_robotics.dynamics import (
    BodyHeadingFrame,
    BodyStateCodec,
    CodecSignature,
    FixedContactFeatureAdapter,
    FixedRigidContactEncoder,
    GravityEncoder,
    JointStateCodec,
    RigidContactSetEncoder,
    expand_rigid_contact_shapes,
)
from physicsnemo_robotics.dynamics.codecs.joint import JointLayout
from physicsnemo_robotics.newton import NewtonStepContext
from tensordict import TensorDict


class JointDynamics:
    """One representation used on stored tensors and, optionally, a live Newton model.

    Applied joint efforts are the sole control input; actuators run outside learned physics.
    Only uniform, joint-backed rigid worlds are supported. Contact inputs are fixed plane probes
    or live Newton rigid contact sets, recomputed from each predicted state during deployment.
    The descriptor is JSON data, so offline training does not initialize a simulation.
    """

    def __init__(
        self,
        descriptor=None,
        model=None,
        *,
        frame_dt=None,
        heading=False,
        contact_bodies=(),
        root_body="",
        max_contacts=0,
    ):
        if descriptor is None:
            if max_contacts < 0 or (max_contacts and contact_bodies):
                raise ValueError("Select fixed contact_bodies or a positive Newton max_contacts capacity.")
            if (
                model.particle_count
                or model.body_count != model.joint_count
                or model.articulation_count != model.world_count
            ):
                raise ValueError("NeRD requires one complete joint-backed rigid articulation per world.")
            self.codec = JointStateCodec(model, heading=heading)
            descriptor = {
                "version": 2,
                "frame_dt": frame_dt,
                "control": "joint_f",
                "heading": heading,
                "contact_bodies": list(contact_bodies),
                "root_body": root_body,
                "max_contacts": max_contacts,
                "contact_entities": [
                    model.body_label[i].rsplit("/", 1)[-1] for i in np.flatnonzero(model.body_world.numpy() == 0)
                ]
                if max_contacts
                else [],
                "codec": self.codec.signature.to_dict(),
                "model": model_signature(model, contact_conditioned=bool(max_contacts)),
            }
        else:
            if descriptor["version"] != 2:
                raise ValueError("NeRD requires descriptor version 2; recollect data with complete physical identity.")
            if descriptor["control"] != "joint_f":
                raise ValueError("NeRD requires applied joint efforts; recollect and refit target-conditioned data.")
            if model is not None:
                self.codec = JointStateCodec(model, heading=descriptor["heading"])
            elif "codec" in descriptor:
                self.codec = JointStateCodec.from_signature(CodecSignature.from_dict(descriptor["codec"]))
            else:
                # Existing raw v2 recordings precede portable codec signatures. Their
                # joint layout still describes the stored world-space state exactly.
                layout = dict(descriptor["layout"])
                for name in ("continuous_q_mask", "root_translation_mask"):
                    layout[name] = torch.tensor(layout[name], dtype=torch.bool)
                for name in ("quaternion_q_starts", "free_root_translation_q_starts"):
                    layout[name] = tuple(layout[name])
                layout["world_count"] = 1
                self.codec = JointStateCodec(layout=JointLayout(**layout), heading=descriptor["heading"])
        if self.codec.heading != descriptor["heading"] or (
            "codec" in descriptor
            and self.codec.signature.fingerprint != CodecSignature.from_dict(descriptor["codec"]).fingerprint
        ):
            raise ValueError("NeRD codec signature differs from the recorded state/heading convention.")
        self.descriptor = descriptor
        self.gravity = GravityEncoder(self.codec, (0.0, 0.0, -1.0)) if descriptor["heading"] else None
        self.max_contacts = descriptor.get("max_contacts", 0)
        if model is not None:
            if (
                np.any(model.joint_target_ke.numpy())
                or np.any(model.joint_target_kd.numpy())
                or model.custom_frequency_counts.get("mujoco:actuator", 0)
            ):
                raise ValueError(
                    "NeRD requires all motor effort in joint_f; solver-side drives are unsupported. "
                    "Compute and clip efforts in the actuator before each physics step."
                )
            self._input_valid = torch.empty((), dtype=torch.bool, device=model.device.alias)
        self.contact_encoder = self.contact_adapter = None
        if model is not None and self.max_contacts:
            body_worlds = model.body_world.numpy()
            indices = np.stack([np.flatnonzero(body_worlds == i) for i in range(model.world_count)])
            names = [model.body_label[i].rsplit("/", 1)[-1] for i in indices[0]]
            if names != descriptor["contact_entities"]:
                raise ValueError("NeRD contact body order differs from the recorded representation.")
            body_codec = BodyStateCodec(
                model,
                indices=indices,
                semantic_ids=tuple(names),
                model_frame=BodyHeadingFrame(body=names.index(descriptor["root_body"]))
                if descriptor["heading"]
                else None,
            )
            self.contact_encoder = RigidContactSetEncoder(model, body_codec, max_contacts=self.max_contacts)
        elif model is not None and descriptor["contact_bodies"]:
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
        return self.codec.layout.dof_qd

    @property
    def contact_dim(self):
        return 14 * len(self.descriptor["contact_bodies"])

    @property
    def contact_shape(self):
        return (self.max_contacts, 17) if self.max_contacts else (self.contact_dim,)

    @property
    def contact_model_cfg(self):
        if not self.max_contacts:
            return {}
        return dict(
            contact_dim=17,
            max_contacts=self.max_contacts,
            num_contact_entities=len(self.descriptor["contact_entities"]),
        )

    @property
    def input_dim(self):
        return (
            self.codec.state_shape[0] + self.control_dim + (3 if self.descriptor["heading"] else 0) + self.contact_dim
        )

    def inputs(self, state, control, contacts):
        """Encode one frame or a batch of sequences without discarding autograd."""
        values = [self.codec.encode_state(state), control]
        if self.descriptor["heading"]:
            values.append(self.gravity.encode_window(TensorDict({"history/state": state}, batch_size=[])))
        if not self.max_contacts:
            values.append(contacts)
        return torch.cat(values, dim=-1)

    def contacts(self, state, contacts=None):
        """Encode starting-state geometry; variable contacts must come from the live collision pipeline."""
        if self.max_contacts:
            dt = self.descriptor["frame_dt"]
            context = NewtonStepContext(state, state, None, contacts, 0, 0, dt, dt)
            values = self.contact_encoder.encode_step(context)
            torch._assert_async((self.contact_encoder.last_overflow == 0).all(), "NeRD contact capacity overflowed.")
            return values
        if self.contact_encoder is None:
            return torch.empty(self.codec.batch_size, 0, device=state.joint_q.device.alias)
        return self.contact_adapter.encode_tokens(self.contact_encoder.encode_state(state))

    def check_inputs(self, state):
        """Reject unrepresented inputs at the same solver boundary during capture and deployment."""
        # Persistent reduction outputs also work inside the numerical teacher's Warp CUDA graph.
        device = state.body_f.device
        stream = torch.cuda.stream(wp.stream_to_torch(wp.get_stream(device))) if device.is_cuda else nullcontext()
        with stream:
            torch.any(wp.to_torch(state.body_f), out=self._input_valid)
            torch._assert_async(self._input_valid.logical_not_(), "NeRD has no external body-wrench input.")


def model_signature(model: newton.Model, *, contact_conditioned: bool = False) -> str:
    """Fingerprint the supported rigid model without mutating it or including solver/viewer attributes.

    Replicated worlds must have the same physical parameters. Local transform roundoff from USD
    placement is tolerated at 1e-6; masses, gains, limits and friction remain exact. Horizontal
    translation is ignored only in scenes with no static geometry except horizontal planes.
    Fixed-root orientation and height remain physical parameters. Mesh geometry is included by content.
    Live-contact models instead fix robot geometry and uniform external material, allowing static
    terrain geometry and floating-root placement to vary through the encoded state and contacts.
    """
    digest = hashlib.sha256()
    collision_shapes = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    shape_types, shape_bodies = model.shape_type.numpy(), model.shape_body.numpy()
    transforms = model.shape_transform.numpy()
    static = collision_shapes & (shape_bodies == -1)
    horizontal = (shape_types == newton.GeoType.PLANE) & np.all(np.abs(transforms[:, 3:5]) < 1.0e-8, axis=1)
    translation_invariant = np.all(horizontal[static])
    if contact_conditioned:
        # Geometry is a live input. Its contact law is not: every external surface must share it.
        for name in (
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
        ):
            values = getattr(model, "shape_" + name).numpy()[static]
            if not len(values) or not np.all(values == values[0]):
                raise ValueError("Contact-conditioned NeRD requires one uniform external contact material.")
            digest.update(name.encode())
            digest.update(np.asarray(values[0]).tobytes())
        collision_shapes &= ~static
    groups = {
        "body": ("mass", "inertia", "com", "flags"),
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
            if name == "joint_X_p" and translation_invariant:
                array[model.joint_parent.numpy() == -1, :2] = 0.0
            if name == "joint_X_p" and contact_conditioned:
                roots = (model.joint_parent.numpy() == -1) & (model.joint_type.numpy() == newton.JointType.FREE)
                array[roots, :3] = 0.0
            if name == "shape_transform":
                # A horizontal plane is unchanged by translation in its own plane.
                array[horizontal, :2] = 0.0
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
    body_worlds = model.body_world.numpy()
    body_ids = np.argsort(body_worlds, kind="stable").reshape(model.world_count, -1)
    body_slots = np.empty(model.body_count, dtype=np.int64)
    body_slots[body_ids] = np.arange(body_ids.shape[1])
    for name, values in (
        ("joint_parent", model.joint_parent.numpy()),
        ("joint_child", model.joint_child.numpy()),
        ("shape_body", model.shape_body.numpy()),
    ):
        entity = "shape" if name == "shape_body" else "joint"
        worlds = getattr(model, entity + "_world").numpy()
        selected = worlds >= 0
        if entity == "shape":
            selected &= collision_shapes
        indices = np.flatnonzero(selected)
        indices = indices[np.argsort(worlds[indices], kind="stable")].reshape(model.world_count, -1)
        values = np.where(values >= 0, body_slots[values.clip(min=0)], -1)
        local = values[indices]
        if not np.all(local == local[0]):
            raise ValueError(f"NeRD cannot represent per-world variation in {name}.")
        digest.update(name.encode())
        digest.update(local[0].tobytes())
    for shape in np.flatnonzero((model.shape_world.numpy() == 0) & collision_shapes):
        source = model.shape_source[shape]
        if source is not None:
            if not isinstance(source, newton.Mesh):
                raise ValueError("NeRD supports primitive and mesh collision geometry only.")
            digest.update(np.asarray(source.vertices).tobytes())
            digest.update(np.asarray(source.indices).tobytes())
    # Pair indices include visual shapes and all replicas. Normalize them to collision-only
    # slots so headless collection and playback with a different world count have the same identity.
    shape_world = model.shape_world.numpy()
    local_shapes = np.flatnonzero(collision_shapes & (shape_world >= 0))
    local_shapes = local_shapes[np.argsort(shape_world[local_shapes], kind="stable")].reshape(model.world_count, -1)
    slots = np.full(model.shape_count, -1, dtype=np.int64)
    slots[local_shapes] = np.arange(local_shapes.shape[1])
    global_shapes = np.flatnonzero(collision_shapes & (shape_world == -1))
    slots[global_shapes] = -1 - np.arange(len(global_shapes))
    for name, pairs in (
        ("excluded_pairs", model.shape_collision_filter_pairs_array()),
        ("contact_pairs", model.shape_contact_pairs.numpy()),
    ):
        pairs = np.asarray(pairs).reshape(-1, 2)
        pairs = pairs[collision_shapes[pairs].all(axis=1)]
        pair_worlds = shape_world[pairs]
        # Different local worlds cannot collide; a global shape can participate in each world.
        pairs = pairs[(pair_worlds[:, 0] == pair_worlds[:, 1]) | (pair_worlds == -1).any(axis=1)]
        worlds = shape_world[pairs].max(axis=1)
        canonical = np.sort(slots[pairs], axis=1)
        order = np.lexsort((canonical[:, 1], canonical[:, 0], worlds))
        canonical, worlds = canonical[order], worlds[order]
        counts = np.bincount(worlds[worlds >= 0], minlength=model.world_count)
        if np.any(counts != counts[0]):
            raise ValueError(f"NeRD cannot represent per-world variation in {name}.")
        local = canonical[worlds >= 0].reshape(model.world_count, -1, 2)
        if not np.all(local == local[0]):
            raise ValueError(f"NeRD cannot represent per-world variation in {name}.")
        digest.update(name.encode())
        digest.update(local[0].tobytes())
        digest.update(canonical[worlds == -1].tobytes())
    gravity = model.gravity.numpy()
    if not np.all(gravity == gravity[0]):
        raise ValueError("NeRD cannot represent per-world gravity variation.")
    digest.update(np.asarray(gravity[0]).tobytes())
    return digest.hexdigest()
