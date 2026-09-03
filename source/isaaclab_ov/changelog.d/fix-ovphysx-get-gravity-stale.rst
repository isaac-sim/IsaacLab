Fixed
^^^^^

* Fixed :meth:`~isaaclab_ov.physics.OvPhysxManager.get_gravity` returning the construction-time
  gravity after :meth:`~isaaclab_ov.physics.OvPhysxManager.set_gravity` changed the running scene.
  The manager now tracks the applied gravity vector, while ``SimulationCfg.gravity`` stays the
  nominal value that randomization terms resample from.
