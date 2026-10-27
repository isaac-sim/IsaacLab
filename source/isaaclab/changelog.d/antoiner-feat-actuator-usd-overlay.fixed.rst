* Fixed Newton-native actuator authoring leaving a joint's ``NewtonActuator`` prims deactivated when a
  configuration was rejected partway through authoring.
* Fixed Newton-native actuator authoring deactivating actuator prims whose first ``newton:targets`` target
  was not covered by a configuration. Newton drives only the first target.
* Fixed fractional Newton-native actuator overrides being truncated when the USD actuator authored
  an integer-typed attribute, while keeping the asset's weaker-layer opinions unchanged.
* Fixed Newton-native actuator authoring skipping validation hooks on supported actuator config subclasses.
