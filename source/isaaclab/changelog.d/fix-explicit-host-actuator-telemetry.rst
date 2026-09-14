Fixed
^^^^^

* Avoided unnecessary joint stiffness, damping, and effort-limit reads when
  updating Newton-native actuator telemetry for all-explicit PhysX and OVPhysX
  articulations, restoring OVPhysX stepping performance while preserving
  computed and applied effort telemetry.
