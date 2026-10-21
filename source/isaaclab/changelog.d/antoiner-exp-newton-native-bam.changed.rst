* Changed :class:`~isaaclab.actuators.BamActuatorCfg` authoring to Newton's native BAM composition:
  ``NewtonBAMControlAPI``, ``NewtonMaxEffortClampingAPI``, a ``NewtonActuatorDelayAPI`` buffer sized by
  ``max_delay``, and a backlash input processor bound through ``newton:backlashJoint``. The stiff friction
  constraint is authored as the servo joints' ``mjc:solreffriction`` and ``mjc:solimpfriction``. Gearbox
  friction, supply sag and the command lag are no longer authored; Isaac Lab runs them from the configuration.
* Changed :class:`~isaaclab.actuators.newton.NewtonActuatorAdapter` to build Newton actuators with
  :class:`~newton.actuators.InputProcessorDelay` instead of the deprecated ``Delay``.
