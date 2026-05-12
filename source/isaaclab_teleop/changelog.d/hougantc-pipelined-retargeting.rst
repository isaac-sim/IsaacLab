Added
^^^^^

* Added :attr:`~isaaclab_teleop.IsaacTeleopCfg.retargeting_execution` for
  configuring IsaacTeleop retargeting execution mode from Isaac Lab.

Changed
^^^^^^^

* Changed :class:`~isaaclab_teleop.IsaacTeleopCfg` to enable IsaacTeleop
  deadline-paced pipelined retargeting by default. This returns the latest
  completed retargeting output while the current frame is submitted, using
  ``DeadlinePacingConfig(safety_margin_s=0.025)`` to sample close to the next
  simulation consumption point and stagger IsaacTeleop's Python work behind
  Isaac Lab's step Python. Set
  ``retargeting_execution=RetargetingExecutionConfig(mode="sync")`` to restore
  exact current-frame retargeting.

Fixed
^^^^^

* Fixed installation to upgrade to the latest compatible ``isaacteleop``
  package when installing ``isaaclab_teleop``.
