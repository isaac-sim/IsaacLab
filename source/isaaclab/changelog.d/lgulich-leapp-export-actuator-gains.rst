Fixed
^^^^^

* Fixed LEAPP export baking zero ``kp``/``kd`` gains into the policy graph for policies trained
  with explicit actuators (e.g. :class:`~isaaclab.actuators.DCMotor`,
  :class:`~isaaclab.actuators.IdealPDActuator`). Such actuators compute their PD term internally
  and apply it as joint effort, leaving the simulation-level joint stiffness/damping at zero, so
  the exporter -- which read ``data.default_joint_stiffness`` / ``data.default_joint_damping`` --
  exported zero gains and the deployed policy was unactuated. The gain outputs are now sourced from
  ``asset.actuators`` so the exported graph carries the real PD gains for every actuator model.
