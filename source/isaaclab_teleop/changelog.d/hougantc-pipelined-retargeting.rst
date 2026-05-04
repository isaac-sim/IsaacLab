Added
^^^^^

* Added :attr:`~isaaclab_teleop.IsaacTeleopCfg.retargeting_execution` for
  configuring IsaacTeleop retargeting execution mode from Isaac Lab.

Changed
^^^^^^^

* Changed :class:`~isaaclab_teleop.IsaacTeleopCfg` to enable IsaacTeleop
  pipelined retargeting by default when supported by the installed
  IsaacTeleop version. Set
  ``retargeting_execution=RetargetingExecutionConfig(mode="sync")`` to restore
  exact current-frame retargeting.
