* Fixed :class:`~isaaclab_ov.sensors.ContactSensor`, :class:`~isaaclab_ov.sensors.JointWrenchSensor` and
  :class:`~isaaclab_ov.sensors.FrameTransformer` resetting the wrong environments, pairing frames of different
  environments, or failing with a CUDA device-side assert when their asset exists in only some environments, for
  example through :attr:`~isaaclab.cloner.CloneCfg.clone_combinations`. They now raise a :class:`RuntimeError` at
  initialization.
