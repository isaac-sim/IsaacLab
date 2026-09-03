# PhysX "true fix" experiment: randomize friction while PRESERVING compliant contact,
# via RigidBodyView.set_compliant_material_properties (4-tuple: fric,fric,stiffness,damping)
# instead of set_material_properties (3-tuple rigid, which wipes compliant).
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from isaaclab.managers.manager_base import ManagerTermBase


class RandomizeFrictionKeepCompliant(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        self._sr = tuple(cfg.params.get("static_friction_range", (0.5, 1.0)))
        self._dr = tuple(cfg.params.get("dynamic_friction_range", (0.5, 1.0)))
        self._logged = False

    def __call__(self, env, env_ids, static_friction_range=None, dynamic_friction_range=None,
                 restitution_range=None, num_buckets=None, asset_cfg=None, make_consistent=True):
        view = self.asset.root_view
        count = int(view.count)
        max_shapes = int(view.max_shapes)
        comp_wp, _combo_wp = view.get_compliant_material_properties()
        comp = wp.to_torch(comp_wp)  # (count, max_shapes, 2) = [stiffness, damping]
        device = comp.device
        stiff = comp[..., 0].clone().float()
        damp = comp[..., 1].clone().float()
        if float(stiff.max().item()) < 1.0:  # fallback to known spawn compliant
            stiff[:] = 300.0
            damp[:] = 30.0
        static = torch.empty((count, max_shapes), device=device).uniform_(self._sr[0], self._sr[1])
        dynamic = torch.empty((count, max_shapes), device=device).uniform_(self._dr[0], self._dr[1])
        dynamic = torch.minimum(static, dynamic)
        data = torch.stack([static, dynamic, stiff, damp], dim=-1).contiguous().float()
        combo = torch.zeros((count, 3), dtype=torch.uint8, device=device).contiguous()
        indices = wp.array(np.arange(count, dtype=np.uint32), dtype=wp.uint32, device=str(device))
        view.set_compliant_material_properties(
            wp.from_torch(data, dtype=wp.float32),
            wp.from_torch(combo, dtype=wp.uint8),
            indices,
        )
        if not self._logged:
            self._logged = True
            print(f"[TRUEFIX] keep-compliant DR applied: count={count} shapes={max_shapes} "
                  f"stiff~{float(stiff.mean()):.1f} damp~{float(damp.mean()):.1f} "
                  f"fric[{self._sr[0]},{self._sr[1]}]", flush=True)
