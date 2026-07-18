Added
^^^^^

* Added :func:`~isaaclab.utils.assets.search_simready_usd_paths` to resolve USD asset paths from a
  SimReady USD-Search semantic query, and the :mod:`isaaclab.sim.spawners.simready` sub-module with
  :class:`~isaaclab.sim.spawners.SimReadyUsdFileCfg` and :class:`~isaaclab.sim.spawners.SimReadyMultiUsdFileCfg`
  to spawn the resolved assets instead of hardcoding USD file paths. The required ``simready-search``
  package is available through the optional ``simready`` extra (``./isaaclab.sh -i simready``).
