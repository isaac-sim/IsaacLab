* **Breaking:** Removed ``isaaclab_newton.actuators.DriveBam`` in favor of Newton's native BAM composition.
  Configure BAM servos through :class:`~isaaclab.actuators.BamActuatorCfg` as before. Its ``kp_scale`` and
  ``kd_scale`` have no replacement; write the native drive's ``kp`` (``kp_fw * error_gain``) through the
  ``"drive"`` component instead. Read and write the friction scale and the supply through
  :func:`~isaaclab_newton.actuators.get_bam_group` instead of the drive's ``friction_scale``, ``vin`` and
  ``sag_gain``.
* **Breaking:** Removed ``isaaclab_newton.physics.MjWarpActuatorBridge``. BAM friction reads the previous solve's
  loads from Newton's MuJoCo solver observables and publishes its budget as the servos' joint friction. Read the
  live budget through :func:`~isaaclab_newton.actuators.get_bam_group` (``friction_budget``) or
  ``data.joint_friction`` instead.
