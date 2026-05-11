Changed
^^^^^^^

* Split :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` construction
  into a pre-physics ``__init__`` (stores cfg and registers the Newton-Warp
  scene-data requirement on
  :class:`~isaaclab.sim.SimulationContext`) and a post-physics
  :meth:`~isaaclab_newton.renderers.NewtonWarpRenderer.initialize` (reads
  the built Newton model.
