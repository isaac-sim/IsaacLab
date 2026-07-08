Added
^^^^^

* Added OVPhysX backend support to :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material`.
  OVPhysX runs the PhysX solver, so materials are bucket-sampled (PhysX's 64000-material limit
  applies) and written per shape through the asset's
  :class:`~isaaclab_ovphysx.sim.views.OvPhysxView`. Randomizes all shapes; per-body selection via
  ``asset_cfg.body_ids`` is not supported, as the ovphysx wheel exposes no per-body shape counts.
