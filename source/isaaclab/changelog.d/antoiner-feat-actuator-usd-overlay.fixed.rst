* Fixed Newton-native actuator authoring leaving a joint's ``NewtonActuator`` prims deactivated when a
  configuration was rejected partway through authoring.
* Fixed Newton-native actuator authoring deactivating actuator prims whose first ``newton:targets``
  target was not covered by a configuration. Newton drives only the first target.
