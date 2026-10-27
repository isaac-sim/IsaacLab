* **Breaking:** :class:`~isaaclab.physics.PhysicsManager` takes its physics configuration at construction, and
  :class:`~isaaclab.sim.SimulationContext` builds the manager with ``instantiate(cfg.physics)``. Subclasses accept
  ``cfg`` in ``__init__`` and pass it to ``super().__init__(cfg)``; ``initialize()`` no longer sets ``_cfg``.
