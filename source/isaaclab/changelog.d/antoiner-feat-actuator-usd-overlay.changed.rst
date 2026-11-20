* **Breaking:** Changed Newton-native actuator authoring to write an explicit actuator configuration onto
  a joint's USD ``NewtonActuator`` prim with the same drive instead of replacing it, so values left unset
  keep the asset's value instead of code defaults such as zero gains. Set a value explicitly to restore the
  previous result; see "Unset values in explicit actuator configurations" in the migration guide.
* **Breaking:** Changed the :attr:`~isaaclab.actuators.DelayedPDActuatorCfg.max_delay` default from 0 to
  None, which keeps a USD actuator's delay with native actuators. Set ``max_delay=0`` to remove it.
* **Breaking:** Changed the :class:`~isaaclab.actuators.BamActuatorCfg` and :class:`~isaaclab.actuators.BamMotorCfg`
  fields to default to None, which keeps the USD BAM actuator's value, and stopped resetting its runtime
  scales, sag gain, and delay seed, which now start from the asset's values. Set a field explicitly to
  restore its previous default.
* **Breaking:** Changed Newton-native actuator authoring to resolve ``saturation_effort`` and
  ``actuator_velocity_limit`` mappings strictly: unmatched and overlapping patterns now raise ValueError.
  A joint that a ``saturation_effort`` mapping skips no longer gets 0; it keeps its USD actuator's value, or
  raises ValueError when no USD DC motor clamp provides one. Give every joint of the group a value.
* Changed :attr:`~isaaclab.actuators.DCMotorCfg.saturation_effort`, remotized PD lookup tables, and neural
  checkpoints with their MLP settings to accept None, which keeps a USD actuator's value with native
  actuators. New components and Isaac Lab execution still require them.
