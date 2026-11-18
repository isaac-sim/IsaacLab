* Added :attr:`~isaaclab.sim.spawners.from_files.from_files_cfg.FileCfg.make_uninstanceable` to
  :class:`~isaaclab.sim.spawners.from_files.from_files_cfg.UrdfFileCfg` and
  :class:`~isaaclab.sim.spawners.from_files.from_files_cfg.MjcfFileCfg` by moving it from
  :class:`~isaaclab.sim.spawners.from_files.from_files_cfg.UsdFileCfg` to the shared base. Spawn-time overrides such
  as :attr:`~isaaclab.sim.spawners.from_files.from_files_cfg.FileCfg.collision_props` can then reach the mesh
  colliders that the URDF and MJCF importers author inside USD instances.
