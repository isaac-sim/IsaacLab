* Changed :class:`~isaaclab.actuators.BamActuatorCfg` authoring to Newton's native BAM composition:
  ``NewtonBAMControlAPI`` reading its supply from the control (``newton:supplyFromControl``),
  ``NewtonMaxEffortClampingAPI``, ``NewtonActuatorRandomDelayAPI`` with the lag range, hold probability and
  update period, and ``NewtonActuatorBacklashAPI`` bound through ``newton:backlashJoint``, with
  ``newton:velocityFeedback`` set from :attr:`~isaaclab.actuators.BamActuatorCfg.backlash_velocity_feedback`.
  The stiff friction constraint is authored as the servo joints' ``mjc:solreffriction`` and
  ``mjc:solimpfriction``. Gearbox friction and supply sag are no longer authored; Isaac Lab runs them from
  the configuration.
* Changed :class:`~isaaclab.actuators.newton.NewtonActuatorAdapter` to build Newton actuators with
  :class:`~newton.actuators.InputProcessorDelay` instead of the deprecated ``Delay``, and with
  :class:`~newton.actuators.InputProcessorRandomDelay` from ``NewtonActuatorRandomDelayAPI`` prims.
