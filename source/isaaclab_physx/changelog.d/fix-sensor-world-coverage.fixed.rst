* Fixed :class:`~isaaclab_physx.sensors.Imu`, :class:`~isaaclab_physx.sensors.Pva` and
  :class:`~isaaclab_physx.sensors.JointWrenchSensor` reporting and resetting the wrong environments, and indexing
  past their buffers, when their asset exists in only some environments, for example through
  :attr:`~isaaclab.cloner.CloneCfg.clone_combinations`. They now raise a :class:`RuntimeError` at initialization,
  and :class:`~isaaclab_physx.sensors.FrameTransformer` raises the same error instead of an unrelated
  :class:`ValueError` or :class:`IndexError`.
