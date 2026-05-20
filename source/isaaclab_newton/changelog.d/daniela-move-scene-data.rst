Changed
^^^^^^^

* Updated imports of :class:`~isaaclab.scene_data.SceneDataBackend` and
  :class:`~isaaclab.scene_data.SceneDataFormat` to their new location in
  :mod:`isaaclab.scene_data` (previously :mod:`isaaclab.physics`).

Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`
  retrieving the wrong simulation context. It now uses
  :meth:`~isaaclab.sim.SimulationContext.instance` instead of the stale
  ``PhysicsManager._sim`` reference.
