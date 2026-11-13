* Added :func:`~isaaclab.utils.assets.search_simready_usd_paths` to find SimReady USD assets through the
  SimReady search service by description, bounding-box size along every axis, authored mass, and SimReady
  features. Other request fields of the service pass through unchanged.
* Added :attr:`~isaaclab.sim.spawners.from_files.UsdFileCfg.move_rigid_body_to_root` and
  :func:`~isaaclab.sim.utils.move_rigid_body_to_root` to move a file's single nested rigid body onto
  the spawned prim, so that files nesting their bodies at different paths can be spawned together
  through :class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg`. A body offset from the prim
  keeps its geometry in place, and its pose is read at the prim.
