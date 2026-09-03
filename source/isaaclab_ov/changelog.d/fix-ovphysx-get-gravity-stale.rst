Fixed
^^^^^

* Fixed :meth:`~isaaclab_ov.physics.OvPhysxManager.get_gravity` returning the construction-time
  gravity after :meth:`~isaaclab_ov.physics.OvPhysxManager.set_gravity` changed the running scene.
  The manager now tracks the applied gravity vector, while ``SimulationCfg.gravity`` stays the
  nominal value that randomization terms resample from.
* Fixed :class:`~isaaclab_ov.sensors.Imu` and :class:`~isaaclab_ov.sensors.Pva` reporting gravity
  captured at sensor initialization. Both sensors now re-read the scene gravity on every update, so
  runtime randomization through :func:`~isaaclab.envs.mdp.events.randomize_physics_scene_gravity`
  is reflected in the accelerometer bias and the projected gravity direction.
