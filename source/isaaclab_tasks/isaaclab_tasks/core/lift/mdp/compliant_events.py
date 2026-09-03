# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms that randomize physics materials while preserving PhysX compliant contacts."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from isaaclab.managers.manager_base import ManagerTermBase


class RandomizeFrictionKeepCompliant(ManagerTermBase):
    """Randomize an object's friction while preserving its PhysX compliant-contact material.

    The stock :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` writes the material
    through the rigid tensor API (``RigidBodyView.set_material_properties``: static friction, dynamic
    friction, restitution), which overwrites any compliant-contact spring authored at spawn. For a
    dexterous grasp under PhysX this is fatal: the object reverts to rigid contacts and is knocked out
    of the hand before a grasp can form, so the policy never converges.

    This term randomizes the static/dynamic friction but writes it back through
    ``RigidBodyView.set_compliant_material_properties`` (static friction, dynamic friction, compliant
    stiffness, compliant damping), reading the current stiffness/damping first so the compliant
    contact model survives the randomization. Friction domain randomization is therefore retained
    without sacrificing grasp stability.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        self._static_friction_range = tuple(cfg.params.get("static_friction_range", (0.5, 1.0)))
        self._dynamic_friction_range = tuple(cfg.params.get("dynamic_friction_range", (0.5, 1.0)))

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
        view = self.asset.root_view
        count = int(view.count)
        max_shapes = int(view.max_shapes)
        # read the current compliant material -> (count, max_shapes, 2) = [stiffness, damping]
        compliant, _combine_modes = view.get_compliant_material_properties()
        compliant = wp.to_torch(compliant)
        device = compliant.device
        stiffness = compliant[..., 0].clone().float()
        damping = compliant[..., 1].clone().float()
        # fall back to the authored spawn compliant if the view reports none
        if float(stiffness.max().item()) < 1.0:
            stiffness[:] = 300.0
            damping[:] = 30.0
        # sample new friction, keeping dynamic <= static for physical consistency
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
