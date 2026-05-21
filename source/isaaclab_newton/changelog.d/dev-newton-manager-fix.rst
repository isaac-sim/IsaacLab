Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager._backend_is_newton`
  returning ``False`` when ``PhysicsManager._sim`` was unset but a
  :class:`~isaaclab.sim.SimulationContext` instance existed. The scene-data
  provider lookup now consistently falls back to
  :meth:`~isaaclab.sim.SimulationContext.instance`, via a new
  :meth:`~isaaclab_newton.physics.NewtonManager.get_scene_data_provider`
  helper shared with :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`.
