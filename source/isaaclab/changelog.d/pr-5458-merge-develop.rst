Changed
^^^^^^^

* Changed :func:`~isaaclab.envs.mdp.body_incoming_wrench` to read from
  :class:`~isaaclab.sensors.JointWrenchSensor`. Pass
  ``sensor_cfg=SceneEntityCfg("joint_wrench", body_names=...)`` instead of an
  articulation asset config.

Removed
^^^^^^^

* Removed ``BaseArticulationData.body_incoming_joint_wrench_b``. Add
  :class:`~isaaclab.sensors.JointWrenchSensorCfg` to the scene and read
  :attr:`~isaaclab.sensors.JointWrenchSensorData.force` and
  :attr:`~isaaclab.sensors.JointWrenchSensorData.torque` instead.
