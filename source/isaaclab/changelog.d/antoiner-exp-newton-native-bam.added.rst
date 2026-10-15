* Added :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.add_batteries` and
  :attr:`~isaaclab.actuators.newton.NewtonActuatorAdapter.battery` to power Newton drives, such as
  :class:`~newton.actuators.DriveBAM`, from per-environment :class:`~newton.actuators.Battery` supplies.
* Added :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.register_reset_callback` to reset
  per-environment actuator state kept outside the Newton actuators.
* Added :attr:`~isaaclab.actuators.BamActuatorCfg.backlash_velocity_feedback` to also feed the play-hinge
  velocity back to the BAM firmware and back-EMF through Newton's native backlash processor.
