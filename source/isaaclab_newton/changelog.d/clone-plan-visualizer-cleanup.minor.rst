Removed
^^^^^^^

* **Breaking:** Removed
  ``isaaclab_newton.cloner.newton_replicate.create_newton_visualizer_prebuild_clone_fn``.
  Callers that need a Newton model for visualization should call
  :func:`~isaaclab_newton.cloner.newton_replicate.newton_visualizer_prebuild`
  directly with the ``(sources, destinations, env_ids, mask, positions)`` bundle
  derived from :meth:`~isaaclab.sim.SimulationContext.get_clone_plans`.
