Added
^^^^^

* Added OVPhysX support to the warp frontend: the physics gate now accepts
  :class:`~isaaclab_ovphysx.physics.OvPhysxCfg` in addition to
  :class:`~isaaclab_newton.physics.NewtonCfg`, so warp-frontend tasks run with
  ``presets=ovphysx`` as well as ``presets=newton_mjwarp``.
* Added a backend data-parity test pinning that every ``data.<field>.warp``
  view the warp MDP twins read exists on each warp-capable backend.
