# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms that randomize physics materials while preserving PhysX compliant contacts."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers.manager_base import ManagerTermBase


class RandomizeFrictionKeepCompliant(ManagerTermBase):
    """Randomize an object's friction while preserving its PhysX compliant-contact material.

    Both PhysX and Newton model soft contacts as a Kelvin-Voigt normal spring-damper with the same
    units -- stiffness [N/m], damping [N*s/m]; Newton's USD importer reads PhysX's
    ``compliant_contact_stiffness``/``damping`` as its own ``ke``/``kd`` fallback, so one authored
    material serves both backends. The stiffness is a numerical contact-softness parameter (a
    penalty spring), not a material Young's modulus.

    The stock :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` writes the PhysX
    material through the rigid tensor API (``RigidBodyView.set_material_properties``: static
    friction, dynamic friction, restitution), overwriting the compliant-contact spring authored at
    spawn -- fatal for a dexterous grasp, since the object reverts to rigid contacts.

    On PhysX this term randomizes friction and writes it back through
    ``RigidBodyView.set_compliant_material_properties`` (static friction, dynamic friction,
    compliant stiffness, compliant damping), reading the current stiffness/damping first so the
    compliant model survives. On every other backend (Newton, OVPhysX) it delegates to the stock
    term, whose randomizer is already surgical (it writes only friction bindings and leaves the
    ``ke``/``kd`` contact parameters untouched).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        self._static_friction_range = tuple(cfg.params.get("static_friction_range", (0.5, 1.0)))
        self._dynamic_friction_range = tuple(cfg.params.get("dynamic_friction_range", (0.5, 1.0)))
        # stock randomizer used for non-PhysX backends (Newton / OVPhysX)
        self._stock = randomize_rigid_body_material(cfg, env)

    def _is_physx(self, env) -> bool:
        name = env.sim.physics_manager.__name__.lower()
        return "physx" in name and name != "ovphysxmanager"

    def __call__(
        self,
        env,
        env_ids,
        static_friction_range=(0.5, 1.0),
        dynamic_friction_range=(0.5, 1.0),
        restitution_range=(0.0, 0.0),
        num_buckets=1,
        asset_cfg=None,
        make_consistent=True,
    ):
        # Non-PhysX backends: the stock randomizer preserves compliant (ke/kd) by construction.
        if not self._is_physx(env):
            self._stock(
                env, env_ids, static_friction_range, dynamic_friction_range, restitution_range,
                num_buckets, asset_cfg if asset_cfg is not None else self.asset_cfg, make_consistent,
            )
            return
        view = self.asset.root_view
        count = int(view.count)
        max_shapes = int(view.max_shapes)
        # read the current compliant material -> (count, max_shapes, 2) = [stiffness, damping]
        compliant, _combine_modes = view.get_compliant_material_properties()
        compliant = wp.to_torch(compliant)
        device = compliant.device
        stiffness = compliant[..., 0].clone().float()
        damping = compliant[..., 1].clone().float()
        if float(stiffness.max().item()) < 1.0:  # fall back to the authored spawn compliant
            stiffness[:] = 2500.0
            damping[:] = 100.0
        static = torch.empty((count, max_shapes), device=device).uniform_(self._static_friction_range[0], self._static_friction_range[1])
        dynamic = torch.empty((count, max_shapes), device=device).uniform_(self._dynamic_friction_range[0], self._dynamic_friction_range[1])
        if make_consistent:
            dynamic = torch.minimum(static, dynamic)
        # write back as the compliant 4-tuple: [static friction, dynamic friction, stiffness, damping]
        data = torch.stack([static, dynamic, stiffness, damping], dim=-1).contiguous().float()
        combine = torch.zeros((count, 3), dtype=torch.uint8, device=device).contiguous()
        indices = wp.array(np.arange(count, dtype=np.uint32), dtype=wp.uint32, device=str(device))
        view.set_compliant_material_properties(
            wp.from_torch(data, dtype=wp.float32),
            wp.from_torch(combine, dtype=wp.uint8),
            indices,
        )
