# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in sliding-friction multiplication for MuJoCo Warp contacts."""

import numpy as np
import warp as wp
from mujoco_warp._src.types import MJ_MINMU, vec5
from newton import Contacts, Model, ModelBuilder, State
from newton.solvers import SolverMuJoCo

from pxr import UsdShade


def _read_friction_combine_mode(value, context) -> int | None:
    """Read only authored modes on the collider's bound physics material."""
    material, _ = UsdShade.MaterialBindingAPI(context["prim"]).ComputeBoundMaterial("physics")
    if not material:
        return None
    attr = material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode")
    if not attr or not attr.HasAuthoredValue():
        return None
    # Retain PhysX mode precedence when multiply meets another authored mode.
    return {"average": 0, "min": 1, "multiply": 2, "max": 3}[attr.Get()]


@wp.kernel
def _multiply_contact_friction(
    count: wp.array[int],
    contact_indices: wp.array[int],
    contact_scale: wp.array[float],
    contact_geom: wp.array[wp.vec2i],
    contact_world: wp.array[int],
    contact_friction: wp.array[vec5],
    geom_shape: wp.array2d[int],
    shape_mode: wp.array[int],
    geom_priority: wp.array[int],
    geom_friction: wp.array2d[wp.vec3],
    explicit_pairs: wp.array[wp.int64],
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    cid = tid
    if contact_indices:
        cid = contact_indices[tid]
    if cid < 0 or cid >= contact_friction.shape[0]:
        return
    geoms = contact_geom[cid]
    g1, g2 = geoms[0], geoms[1]
    # Flex contacts and unequal-priority geoms keep their native parameters.
    if g1 < 0 or g2 < 0:
        return
    if geom_priority[g1] != geom_priority[g2]:
        return
    world = contact_world[cid]
    s1 = geom_shape[world, g1]
    s2 = geom_shape[world, g2]
    if s1 < 0 or s2 < 0:
        return
    if wp.max(shape_mode[s1], shape_mode[s2]) != 2:
        return
    # Explicit MuJoCo pairs own their (potentially anisotropic) friction.
    if explicit_pairs:
        key = wp.int64(wp.min(g1, g2)) * wp.int64(geom_shape.shape[1]) + wp.int64(wp.max(g1, g2))
        index = wp.lower_bound(explicit_pairs, key)
        if index < explicit_pairs.shape[0] and explicit_pairs[index] == key:
            return
    friction_world = world % geom_friction.shape[0]
    mu = wp.max(MJ_MINMU, geom_friction[friction_world, g1][0] * geom_friction[friction_world, g2][0])
    if contact_scale and contact_scale[tid] > 0.0:
        mu *= contact_scale[tid]
    friction = contact_friction[cid]
    friction[0] = mu
    friction[1] = mu
    contact_friction[cid] = friction


def register_friction_combine_mode(builder: ModelBuilder) -> None:
    """Carry authored material modes through USD import and world replication."""
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name="friction_combine_mode",
            namespace="isaaclab",
            frequency=Model.AttributeFrequency.SHAPE,
            assignment=Model.AttributeAssignment.MODEL,
            dtype=wp.int32,
            default=-1,
            usd_attribute_name="*",
            usd_value_transformer=_read_friction_combine_mode,
        )
    )


class MaterialSolverMuJoCo(SolverMuJoCo):
    """Honor explicit material multiplication without changing MuJoCo defaults.

    Only sliding friction is adapted. Other combine modes retain native behavior;
    an explicitly authored ``max`` takes precedence over ``multiply``. MuJoCo geom
    priority, explicit contact pairs, and torsional/rolling friction remain native.
    """

    def __init__(self, model: Model, **kwargs):
        modes = getattr(getattr(model, "isaaclab", None), "friction_combine_mode", None)
        self._multiply_friction = modes is not None and bool(np.any(modes.numpy() == 2))
        if self._multiply_friction and kwargs.get("use_mujoco_cpu", False):
            raise ValueError("Material friction_combine_mode='multiply' requires MuJoCo Warp (use_mujoco_cpu=False).")
        super().__init__(model, **kwargs)
        if self._multiply_friction:
            pairs = np.sort(np.stack((self.mj_model.pair_geom1, self.mj_model.pair_geom2)), axis=0)
            keys = np.sort(pairs[0].astype(np.int64) * self.mj_model.ngeom + pairs[1])
            self._explicit_friction_pairs = wp.array(keys, dtype=wp.int64, device=model.device)
            self.mjw_model.callback.contactfilter = self._update_contact_friction

    def _update_contact_friction(self, model, data, contacts: Contacts | None = None) -> None:
        """Apply the selected rule after collision, before constraint construction."""
        wp.launch(
            _multiply_contact_friction,
            dim=min(contacts.rigid_contact_max, data.naconmax) if contacts is not None else data.naconmax,
            inputs=[
                contacts.rigid_contact_count if contacts is not None else data.nacon,
                self._contact_tid_to_cid if contacts is not None else None,
                contacts.rigid_contact_friction if contacts is not None and contacts.rigid_contact_stiffness else None,
                data.contact.geom,
                data.contact.worldid,
                data.contact.friction,
                self.mjc_geom_to_newton_shape,
                self.model.isaaclab.friction_combine_mode,
                model.geom_priority,
                model.geom_friction,
                self._explicit_friction_pairs if contacts is None else None,
            ],
            device=self.model.device,
        )

    def _convert_contacts_to_mjwarp(self, model: Model, state_in: State, contacts: Contacts):
        super()._convert_contacts_to_mjwarp(model, state_in, contacts)
        if self._multiply_friction:
            # Recompute from geom coefficients so cached contacts do not compound
            # the product across substeps, and runtime material DR stays effective.
            self._update_contact_friction(self.mjw_model, self.mjw_data, contacts)
