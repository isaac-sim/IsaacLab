Added
^^^^^

* Added :attr:`~isaaclab_teleop.IsaacTeleopCfg.retargeting_execution` for
  configuring IsaacTeleop retargeting execution mode from Isaac Lab.

Changed
^^^^^^^

* Changed :class:`~isaaclab_teleop.IsaacTeleopCfg` to enable IsaacTeleop
  deadline-paced pipelined retargeting by default when supported by the
  installed IsaacTeleop version. This returns the latest completed retargeting
  output while the current frame is submitted, using
  ``DeadlinePacingConfig(safety_margin_s=0.025)`` to sample close to
  the next simulation consumption point. Set
  ``retargeting_execution=RetargetingExecutionConfig(mode="sync")`` to restore
  exact current-frame retargeting.
