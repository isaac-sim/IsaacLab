* Added seven externally supplied shoe variants using the default clone assignment strategy and
  uniform visual shape slots for mixed eyelet counts. Single-asset overrides moved from
  ``scene.shoelace_asset.spawn.usd_path`` to a one-element ``scene.shoelace_asset.spawn.assets_cfg`` list.
* Simplified task configuration and read shoe/lace friction from the baked USD materials, retaining
  Franka finger overrides for grasp retention. Custom shoe/lace friction moved from task-level
  ``shoe_mu`` / ``lace_mu`` overrides to the asset materials; finger overrides remained available on
  each robot's ``spawn.friction_overrides``.
