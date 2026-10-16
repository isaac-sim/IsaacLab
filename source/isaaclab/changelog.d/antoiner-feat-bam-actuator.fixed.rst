* Fixed Newton-native actuator authoring leaving a replacement ``NewtonActuator`` prim inactive when it reused
  the path of an actuator prim deactivated during the same call, which left the joint without an actuator.
