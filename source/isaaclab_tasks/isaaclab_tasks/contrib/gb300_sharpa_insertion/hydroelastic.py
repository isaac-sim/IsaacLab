# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydroelastic hand/connector contact with a physically attached VBD cable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab_contrib.coupling import CouplerProxyCfg, CouplerProxyMappingCfg

if TYPE_CHECKING:
    import newton

# Newton exposes this builder attribute but does not publish its mode enum.
_MJWARP_SOLREF_MODE_FORCE_SPACE = 0


def configure_hydroelastic_physics(physics: NewtonCfg) -> None:
    """Keep hydroelastic grasp contacts in MJWarp and the flexible cable in VBD.

    Args:
        physics: Newton configuration whose entries own the rigid scene and cable.
    """
    rigid, cable = physics.solver_cfg.entries
    rigid.bodies.extend((r"{ENV_REGEX_NS}/Cable/head", r"{ENV_REGEX_NS}/Cable/latch"))
    cable.bodies = [r"{ENV_REGEX_NS}/Cable/(?!(?:head|latch)(?:/|$)).*"]
    # The fixture proxy includes the pedestal and rack; only the ground needs static ownership.
    cable.include_static_shapes = False
    cable.shape_label_patterns = [r"/World/GroundPlane/.*"]
    cable.solver_cfg = cable.solver_cfg.replace(
        class_type="isaaclab_tasks.contrib.gb300_sharpa_insertion.endpoint:EndpointVBDSolver"
    )
    physics.solver_cfg = CouplerProxyCfg(
        entries=[rigid, cable],
        # The cable runs its own post-transfer pipeline; the outer pipeline collides rigid pairs once.
        outer_contact_entry="rigid",
        iterations=1,
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="cable",
                bodies=rigid.bodies.copy(),
                mode="staggered",
                collision_pipeline=_one_way_cable_collision_pipeline,
            )
        ],
    )
    physics.collision_cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(
            output_contact_surface=True,
            # A 1 mm² speculative patch; penetrating contacts retain their SDF area.
            margin_contact_area=1.0e-6,
        )
    )
    physics.collision_decimation = 1


def configure_hydroelastic_shapes(
    builder: newton.ModelBuilder,
    grasp_body_patterns: tuple[str, ...],
    *,
    pad_pressure_gradient: float = 5.0e8,
) -> None:
    """Author fingertip/plug SDFs while retaining ordinary contacts for every other pair.

    Massless fingertip copies contact the housing; the thumb also contacts the clip.
    Original colliders retain finger self-contact and contact with the socket.
    Both copies occupy the same surface in the body's local frame [m].
    Ordinary connector/rack contacts use force-space stiffness [N/m] and
    damping [N s/m], without changing the hydroelastic pad response.

    ``pad_pressure_gradient`` controls the compliant covering [N/m^3], without
    changing the rigid connector material or adding mass to the pad copies.
    """
    import newton

    if not np.isfinite(pad_pressure_gradient) or pad_pressure_gradient <= 0.0:
        raise ValueError("pad_pressure_gradient must be finite and positive.")
    solref_mode = builder.custom_attributes.get("mujoco:solref_mode")
    if solref_mode is None:
        raise RuntimeError("MJWarp did not register its per-shape mujoco:solref_mode attribute.")
    if solref_mode.values is None:
        solref_mode.values = {}
    if not isinstance(solref_mode.values, dict):
        raise RuntimeError("MJWarp's per-shape mujoco:solref_mode storage must be sparse.")
    plugs = {
        shape
        for shape, label in enumerate(builder.shape_label)
        if builder.shape_world[shape] == builder.current_world and str(label).endswith("/Cable/head/grip")
    }
    latches = {
        shape
        for shape, label in enumerate(builder.shape_label)
        if builder.shape_world[shape] == builder.current_world and str(label).endswith("/Cable/latch/contact")
    }
    grasp_shapes = []
    thumb_grasp_shapes = set()
    for shape, label in enumerate(tuple(builder.shape_label)):
        if builder.shape_world[shape] != builder.current_world:
            continue
        if not builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        if shape in plugs or str(label).endswith("/Cable/latch/contact") or "/Fixtures/GB300/" in str(label):
            # Both endpoints must opt in; averaging legacy solref values makes
            # the differently damped clip and socket contacts nearly powerless.
            solref_mode.values[shape] = _MJWARP_SOLREF_MODE_FORCE_SPACE
        components = str(label).split("/")
        soft_pad = any(component.endswith("_elastomer") for component in components) and any(
            pattern in str(label) for pattern in grasp_body_patterns
        )
        if soft_pad and any(component.endswith("thumb_elastomer") for component in components):
            solref_mode.values[shape] = _MJWARP_SOLREF_MODE_FORCE_SPACE
        if not (soft_pad or shape in plugs or shape in latches):
            continue
        if soft_pad:
            original = shape
            # A fixed pad is part of its distal link. Inherit that link's
            # authored adjacency exclusions without disabling cross-finger contact.
            pad_body = builder.shape_body[original]
            for mount, joint in builder.joint_parents.get(pad_body, ()):
                if mount < 0 or builder.joint_type[joint] != newton.JointType.FIXED:
                    continue
                if not builder.joint_collision_filter_parent[joint]:
                    continue
                neighbours = (*builder.joint_parents.get(mount, ()), *builder.joint_children.get(mount, ()))
                for neighbour, adjacent_joint in neighbours:
                    if neighbour < 0 or neighbour == pad_body:
                        continue
                    if builder.joint_collision_filter_parent[adjacent_joint]:
                        for other in builder.body_shapes[neighbour]:
                            builder.add_shape_collision_filter_pair(original, other)
            # Separate pair routing without removing ordinary finger self-contact
            # or adding mass and inertia for the coincident collision surface.
            shape = builder.add_shape(
                body=builder.shape_body[original],
                type=builder.shape_type[original],
                xform=builder.shape_transform[original],
                scale=builder.shape_scale[original],
                src=builder.shape_source[original],
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    ke=builder.shape_material_ke[original],
                    kd=builder.shape_material_kd[original],
                    kf=builder.shape_material_kf[original],
                    ka=builder.shape_material_ka[original],
                    mu=builder.shape_material_mu[original],
                    restitution=builder.shape_material_restitution[original],
                    mu_torsional=builder.shape_material_mu_torsional[original],
                    mu_rolling=builder.shape_material_mu_rolling[original],
                    has_particle_collision=False,
                    is_visible=False,
                ),
                label=f"{label}/hydroelastic_grasp",
            )
            grasp_shapes.append(shape)
            for plug in plugs:
                builder.add_shape_collision_filter_pair(original, plug)
            if any(component.endswith("thumb_elastomer") for component in components):
                thumb_grasp_shapes.add(shape)
                for latch in latches:
                    builder.add_shape_collision_filter_pair(original, latch)
        # Native finalization handles scale and shares identical SDFs between pad copies.
        # Keep the connector's already cooked, higher-resolution CAD fields intact.
        sdf_shapes = (original, shape) if soft_pad else (shape,)
        for sdf_shape in sdf_shapes:
            if (
                builder.shape_type[sdf_shape] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH)
                and builder.shape_source[sdf_shape].sdf is not None
            ):
                continue
            builder.shape_sdf_max_resolution[sdf_shape] = 128
            builder.shape_sdf_narrow_band_range[sdf_shape] = (-0.003, 0.003)
            builder.shape_sdf_padding[sdf_shape] = 0.0003
        builder.shape_flags[shape] |= int(newton.ShapeFlags.HYDROELASTIC)
        # Linear pressure gradients [N/m^3]: compliant finger covering and hard
        # connector housing, estimated rather than measured hardware properties.
        builder.shape_material_kh[shape] = pad_pressure_gradient if soft_pad else 1.0e11
        builder.shape_gap[shape] = 0.0003
        builder.shape_margin[shape] = 0.0
    for shape in grasp_shapes:
        for other in range(builder.shape_count):
            thumb_clip = shape in thumb_grasp_shapes and other in latches
            if other != shape and other not in plugs and not thumb_clip:
                builder.add_shape_collision_filter_pair(shape, other)


def _one_way_cable_collision_pipeline(model: newton.Model) -> newton.CollisionPipeline:
    """Allocate only contacts that can affect the cable after proxy transfer."""
    import newton

    # Match Newton's runtime filter once, before sizing narrow-phase buffers.
    # In this view rigid proxies are kinematic; source contacts remain unchanged.
    shape_bodies = model.shape_body.numpy()
    body_flags = model.body_flags.numpy()
    immovable = shape_bodies < 0
    attached = np.flatnonzero(~immovable)
    immovable[attached] = (body_flags[shape_bodies[attached]] & int(newton.BodyFlags.KINEMATIC)) != 0
    pairs = model.shape_contact_pairs.numpy().reshape(-1, 2)
    pairs = pairs[~np.all(immovable[pairs], axis=1)]
    # Override the destination view so optional fields are sized from its usable pairs.
    model.shape_contact_pairs = wp.array(pairs, dtype=wp.vec2i, device=model.device)
    return newton.CollisionPipeline(
        model,
        shape_pairs_filtered=model.shape_contact_pairs,
        contact_matching="sticky",
        include_static_kinematic_pairs=False,
    )
