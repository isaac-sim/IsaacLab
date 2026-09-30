Added
^^^^^

* Rejected :class:`~isaaclab.actuators.BamActuatorCfg` groups with a message naming the required
  native Newton configuration. OVPhysX's native actuator host adapter did not support BAM.
  BAM required the MJWarp solver for solver-hosted friction; other Newton solvers also raised.
  Other supported actuator configurations were unaffected.
