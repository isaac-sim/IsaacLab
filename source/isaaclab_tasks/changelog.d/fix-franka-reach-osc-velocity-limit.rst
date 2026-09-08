Deprecated
^^^^^^^^^^

* Deprecated the ``diffik_abs`` preset on ``Isaac-Reach-Franka-OSC``. The OSC action term replaces the
  arm-controller presets, so on this task the preset only zeroed the action-magnitude reward weight, which
  removed the sole regularizer on the raw pose targets. The preset now resolves as a no-op and emits a
  :class:`FutureWarning`; it will be removed in a future release. Migration: drop ``presets=diffik_abs``
  from ``Isaac-Reach-Franka-OSC`` commands.

Fixed
^^^^^

* Fixed ``Isaac-Reach-Franka-OSC`` dropping the Franka Menagerie solver joint velocity limit. The effort
  actuator copied the deprecated ``velocity_limit_sim`` alias instead of ``joint_velocity_limit``, so the
  arm ran without a velocity clamp and could reach joint speeds that destabilize the simulation.
