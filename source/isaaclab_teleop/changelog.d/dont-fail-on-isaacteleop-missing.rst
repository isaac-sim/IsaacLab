Fixed
^^^^^

* Fixed :class:`~isaaclab_teleop.IsaacTeleopCfg` requiring the optional ``isaacteleop``
  package at import and construction time. Environments that reference it (e.g. the GR1T2
  and Unitree G1 pick-place and locomanipulation tasks) failed to parse with
  ``No module named 'isaacteleop'`` on systems where ``isaacteleop`` is not installed
  (e.g. DGX Spark). The ``isaacteleop`` import is now deferred, and
  :attr:`~isaaclab_teleop.IsaacTeleopCfg.retargeting_execution` defaults to ``None`` and is
  resolved to IsaacTeleop's pipelined, deadline-paced default when a teleop session starts.
