Added
^^^^^

* Added the :mod:`isaaclab.sim.spawners.simready` sub-module with
  :class:`~isaaclab.sim.spawners.SimReadyUsdFileCfg`, :class:`~isaaclab.sim.spawners.SimReadyMultiUsdFileCfg`,
  and :func:`~isaaclab.sim.spawners.search_simready_usd_paths` to resolve spawner USD asset paths from a
  SimReady USD-Search semantic query instead of hardcoded file paths. The required ``simready-search``
  package is available through the optional ``simready`` extra (``./isaaclab.sh -i simready``).
