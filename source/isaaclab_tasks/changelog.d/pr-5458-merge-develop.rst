Changed
^^^^^^^

* Updated classic Ant/Humanoid manager-based environments and direct in-hand
  manipulation environments to read body incoming wrenches from
  :class:`~isaaclab.sensors.JointWrenchSensor` instead of
  ``ArticulationData.body_incoming_joint_wrench_b``. Add a
  :class:`~isaaclab.sensors.JointWrenchSensorCfg` to the scene and pass its
  :class:`~isaaclab.managers.SceneEntityCfg` as ``sensor_cfg``. The classic
  Ant/Humanoid Newton presets now use the same wrench observations as PhysX.
