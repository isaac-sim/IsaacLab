Fixed
^^^^^

* Fixed :attr:`~isaaclab.actuators.ActuatorBaseCfg.friction`,
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.dynamic_friction`, and
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.viscous_friction` docstrings describing the obsolete
  Isaac Sim 4.5 unitless-coefficient model. The fields are documented as efforts
  [N or N·m, depending on joint type] (coefficient [N·s/m or N·m·s/rad] for viscous friction),
  matching measured solver behavior, with a note on the PhysX hard-hold vs Newton soft-constraint
  difference.
