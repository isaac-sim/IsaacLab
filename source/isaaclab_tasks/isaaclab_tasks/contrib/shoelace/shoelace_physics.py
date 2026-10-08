# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construct the shoelace model with its final static physics parameters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.cloner.replicate import newton_builder_world_hook
from newton import GeoType, ModelBuilder
from newton.geometry import compute_inertia_shape

from pxr import Usd

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext

from . import shoelace_constants as physics
from .mdp.events import _startup_array

if TYPE_CHECKING:
    from .shoelace_env_cfg import ShoelaceEnvCfg


def create_shoelace_env(cfg: ShoelaceEnvCfg, **kwargs: object) -> ManagerBasedRLEnv:
    """Create a manager-based environment with task-local builder configuration.

    Args:
        cfg: Shoelace environment configuration, including material and inertia settings.
        **kwargs: Arguments forwarded to the manager-based environment constructor.

    Returns:
        Environment whose static cable physics was configured before solver construction.

    Static properties are authored through the replication builder hook.
    """
    regularization = cfg.cable_inertia_regularization
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("Cable inertia regularization must be finite and nonnegative")

    def configure_world(builder: ModelBuilder, world: int, position: np.ndarray, rotation: np.ndarray) -> None:
        stage = SimulationContext.instance().stage
        configure_shoelace_builder(builder, stage, world, rotation, cfg)

    with newton_builder_world_hook(configure_world):
        return ManagerBasedRLEnv(cfg=cfg, **kwargs)


def configure_shoelace_builder(
    builder: ModelBuilder, stage: Usd.Stage, world: int, rotation: np.ndarray, cfg: ShoelaceEnvCfg
) -> None:
    """Configure the just-appended world's cable bodies, joints, and contact shapes.

    Applies four task-specific physics features before model finalization:

    * Set cable contact friction, stiffness, damping, and gap [m].
    * Recompute capsule inertia and add isotropic regularization [kg*m^2].
    * Zero mass, inertia, and their inverses for cable anchors and the kinematic shoe.
    * Set length-scaled rod stiffness/damping with a smoothly stiffened free tip (aglet).

    Args:
        builder: Mutable Newton builder before model allocation.
        stage: Stage containing the retained source environment.
        world: Appended world's index.
        rotation: Unused world placement quaternion; imported body orientations are preserved.
        cfg: Task settings; inertia regularization is in [kg*m^2].
    """
    # Hooks run immediately after appending a world, so only visit its new bodies/joints.
    bodies = []
    for body in range(builder.body_count - 1, -1, -1):
        if builder.body_world[body] != world:
            break
        bodies.append(body)
    joints = []
    for joint in range(builder.joint_count - 1, -1, -1):
        if builder.joint_world[joint] != world:
            break
        joints.append(joint)

    for side, free_end_at_start in (("Left", True), ("Right", False)):
        suffix = f"/ShoelaceScene/Shoelace{side}/geometry/mesh"
        curve = stage.GetPrimAtPath(f"/World/envs/env_0{suffix}")
        # Numeric suffixes recover curve order; lexical sorting would put segment 10 before segment 2.
        chain = sorted(
            (body for body in bodies if f"{suffix}_edge_body_" in builder.body_label[body]),
            key=lambda body: int(builder.body_label[body].rsplit("_", 1)[-1]),
        )
        rod_joints = sorted(
            (joint for joint in joints if f"{suffix}_cable_" in builder.joint_label[joint]),
            key=lambda joint: int(builder.joint_label[joint].rsplit("_", 1)[-1]),
        )
        if not chain or len(rod_joints) != len(chain) - 1:
            raise ValueError(f"Expected one open cable chain for {side} in world {world}")
        # Preserve the material tuning when the asset is resampled to a different segment length [m].
        reference_length = float(_startup_array(curve, "referenceSegmentLength", ()))
        segment_length = float(_startup_array(curve, "segmentLength", ()))
        if reference_length <= 0.0 or segment_length <= 0.0:
            raise ValueError(f"{curve.GetPath()}: segment lengths must be positive")
        # The anchored end is opposite the free tail: left cable last segment, right cable first.
        anchor = chain[-1 if free_end_at_start else 0]
        for body in chain:
            shapes = builder.body_shapes[body]
            if len(shapes) != 1 or builder.shape_type[shapes[0]] != GeoType.CAPSULE:
                raise ValueError("Shoelace construction requires one capsule shape per segment")
            # Configure generated capsule contacts before solver views exist; no runtime notification is needed.
            shape = shapes[0]
            builder.shape_material_mu[shape] = cfg.lace_mu
            builder.shape_material_ke[shape] = physics.CONTACT_KE
            builder.shape_material_kd[shape] = physics.CONTACT_KD
            builder.shape_gap[shape] = physics.CONTACT_GAP
            # Scale unit-density capsule inertia to the imported mass, then add isotropic regularization.
            unit_mass, _, unit_inertia = compute_inertia_shape(
                GeoType.CAPSULE, builder.shape_scale[shape], None, density=1.0
            )
            inertia = builder.body_mass[body] * np.asarray(unit_inertia).reshape(3, 3) / unit_mass
            inertia += np.eye(3) * cfg.cable_inertia_regularization
            # Zero inverse mass/inertia fixes the anchor without adding joints or changing body flags.
            if body == anchor:
                builder.body_mass[body] = 0.0
                inertia.fill(0.0)
            # Keep inverse quantities consistent with the modified mass and inertia.
            builder.body_inertia[body] = wp.mat33(inertia)
            builder.body_inv_mass[body] = 0.0 if body == anchor else 1.0 / builder.body_mass[body]
            builder.body_inv_inertia[body] = wp.mat33(0.0) if body == anchor else wp.mat33(np.linalg.inv(inertia))
        # Model a stiff shoelace tip (aglet) to help prevent the free tail from lying flush against
        # the shoe surface and becoming ungraspable. Blend smoothly into the flexible cable;
        # reverse the profile for the right cable.
        weights = _tail_joint_blend_weights(len(rod_joints), segment_length, free_end_at_start)
        for target, stretch, bend, tail in (
            (builder.joint_target_ke, physics.STRETCH_STIFFNESS, physics.BEND_STIFFNESS, physics.TAIL_BEND_STIFFNESS),
            (builder.joint_target_kd, physics.STRETCH_DAMPING, physics.BEND_DAMPING, physics.TAIL_BEND_DAMPING),
        ):
            values = np.column_stack((np.full_like(weights, stretch), bend + weights * (tail - bend)))
            # Match the rod's four gain slots: two stretch/shear entries and two bend/twist entries.
            values = np.repeat(values * reference_length / segment_length, 2, axis=1)
            for joint, gains in zip(rod_joints, values, strict=True):
                start = builder.joint_qd_start[joint]
                target[start : start + 4] = gains.tolist()
    # Preserve the shoe's zero-mass contact model while retaining its imported kinematic flag.
    for body in bodies:
        if builder.body_label[body].endswith("/ShoelaceScene/Shoe"):
            builder.body_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_mass[body] = 0.0
            builder.body_inv_inertia[body] = wp.mat33(0.0)


def _tail_joint_blend_weights(
    joint_count: int,
    segment_length: float,
    free_end_at_start: bool,
) -> np.ndarray:
    """Return smooth weights for the distal tail's runtime material profile.

    Args:
        joint_count: Number of joints in the cable.
        segment_length: Mean resampled segment length [m].
        free_end_at_start: Whether the free end precedes the fixed end in joint order.

    Returns:
        Dimensionless weights in joint order, shape [joint_count].
    """
    distances = segment_length * np.arange(1, joint_count + 1, dtype=np.float64)
    if physics.TAIL_STIFF_TRANSITION_LENGTH > 0.0:
        weights = np.clip(
            (physics.TAIL_STIFF_CORE_LENGTH + physics.TAIL_STIFF_TRANSITION_LENGTH - distances)
            / physics.TAIL_STIFF_TRANSITION_LENGTH,
            0.0,
            1.0,
        )
        weights = weights * weights * (3.0 - 2.0 * weights)
    else:
        weights = (distances <= physics.TAIL_STIFF_CORE_LENGTH).astype(np.float64)
    return weights if free_end_at_start else weights[::-1].copy()
