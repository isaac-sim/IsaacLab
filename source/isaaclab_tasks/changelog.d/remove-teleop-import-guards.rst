Changed
^^^^^^^

* Removed the per-environment ``try/except ImportError`` guards around the
  ``isaacteleop`` / ``isaaclab_teleop`` imports in the Galbot, Franka, GR1T2
  nut-pour, and GR1T2 exhaust-pipe task configs. The imports are now
  unconditional, matching the other teleop task configs, now that
  :class:`~isaaclab_teleop.IsaacTeleopCfg` no longer requires the optional
  ``isaacteleop`` package at import time. No migration is needed:
  ``isaaclab_teleop`` ships with Isaac Lab and teleoperation behavior is
  unchanged.
