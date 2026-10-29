* Added the ``batteries`` argument of :class:`~isaaclab.actuators.newton.NewtonActuatorAdapter`, which steps the
  batteries that power Newton motors, such as Newton's ``model.batteries``, after the actuators and restores the
  batteries of the reset environments. Newton does not step its batteries itself.
* Added :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.register_reset_callback` to reset
  per-environment actuator state kept outside the Newton actuators.
* Added the ``"motor"`` component to :func:`~isaaclab.actuators.newton.read_group_parameter` and
  :func:`~isaaclab.actuators.newton.write_group_parameter`, for actuators with a motor such as
  :class:`~newton.actuators.MotorDC`.
* Added :attr:`~isaaclab.actuators.BamActuatorCfg.battery_resistance` and
  :attr:`~isaaclab.actuators.BamActuatorCfg.battery_resistance_range` [Ohm], the internal resistance of the
  battery each BAM group draws from.
