* Added :func:`~isaaclab.utils.assets.search_simready_usd_paths` to resolve USD asset paths from a
  SimReady USD-Search semantic query. The required ``simready-search`` package is available through
  the optional ``simready`` extra (``uv sync --extra simready``).
* Added :attr:`~isaaclab.sim.spawners.from_files.UsdFileCfg.move_rigid_body_to_root` and
  :func:`~isaaclab.sim.utils.move_rigid_body_to_root` to move a file's single nested rigid body onto
  the spawned prim, so that files nesting their bodies at different paths can be spawned together
  through :class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg`.
