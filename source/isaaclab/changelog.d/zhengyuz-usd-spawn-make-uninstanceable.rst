Added
^^^^^

* Added :attr:`~isaaclab.sim.spawners.from_files.from_files_cfg.UsdFileCfg.make_uninstanceable` to disable
  USD instancing below a spawned prim before overrides are applied. Recursive overrides such as
  :attr:`~isaaclab.sim.spawners.from_files.from_files_cfg.FileCfg.physics_material` can then author
  properties on descendants of an instanceable asset, which are otherwise read-only instance proxies.
