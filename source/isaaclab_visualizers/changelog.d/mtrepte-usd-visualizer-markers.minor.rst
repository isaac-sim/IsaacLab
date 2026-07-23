Added
^^^^^

* Added general USD mesh support for Newton/Rerun/Viser visualization markers via
  :func:`newton.usd.get_mesh`. Any :class:`~isaaclab.sim.spawners.UsdFileCfg` marker
  now loads geometry and material properties (color, texture) directly from the USD file,
  replacing the previous fallback that silently skipped unsupported USD paths.
  The hardcoded DexCube textured-box workaround has been removed.
