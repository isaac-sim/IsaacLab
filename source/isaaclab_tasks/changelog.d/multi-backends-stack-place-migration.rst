Changed
^^^^^^^

* Migrated the Stack-Cube (UR10, Galbot) and Place (Agibot) manipulation tasks to
  Lab 3.0 multi-backend physics by exposing a ``PhysicsCfg`` preset
  (``physics=physx`` / ``physics=newton_mjwarp``) instead of a hard-coded
  :class:`~isaaclab_physx.physics.PhysxCfg`.

Added
^^^^^

* Added a config-time check that raises a clear error when a surface-gripper (suction)
  stacking task is run with the Newton physics backend, which has no surface-gripper
  implementation, instead of failing later during scene creation.
* Added a config-time check that raises a clear error when an Agibot place task is run
  with the Newton physics backend, whose USD parser rejects the robot's reversed
  gripper joints, instead of failing later during scene creation.
