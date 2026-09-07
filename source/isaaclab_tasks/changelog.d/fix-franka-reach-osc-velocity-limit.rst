Fixed
^^^^^

* Fixed ``Isaac-Reach-Franka-OSC`` dropping the Franka Menagerie solver joint velocity limit. The effort
  actuator copied the deprecated ``velocity_limit_sim`` alias instead of ``joint_velocity_limit``, so the
  arm ran without a velocity clamp and could reach joint speeds that destabilize the simulation.
* Fixed ``Isaac-Reach-Franka-OSC`` exposing the ``diffik_abs`` controller preset, which only zeroed the
  action-magnitude reward weight because the OSC action term replaces the arm-controller presets. The
  task now resolves the default reward weight and only exposes physics presets. Migration: drop
  ``presets=diffik_abs`` from ``Isaac-Reach-Franka-OSC`` commands.
