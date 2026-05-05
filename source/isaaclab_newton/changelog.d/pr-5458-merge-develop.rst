Removed
^^^^^^^

* Removed the unimplemented ``ArticulationData.body_incoming_joint_wrench_b``
  accessor. Add :class:`~isaaclab.sensors.JointWrenchSensorCfg` to the scene
  and read :attr:`~isaaclab.sensors.JointWrenchSensorData.force` and
  :attr:`~isaaclab.sensors.JointWrenchSensorData.torque` instead.

Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.JointWrenchSensor` initialization for
  USD assets whose articulation root is nested below the configured asset prim.
