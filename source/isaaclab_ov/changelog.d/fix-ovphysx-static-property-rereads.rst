Fixed
^^^^^

* Fixed OVPhysX articulation joint properties (stiffness, damping, armature, position/velocity/effort
  limits, friction) and body mass/inertia being re-read from their CPU-only bindings on every simulation
  step. These static properties are now read once after invalidation and kept current by the
  ``write_*`` / ``set_*`` setters. This removes three blocking host round-trips per physics step from the
  native Newton actuator path, which made it several times slower than the Isaac Lab actuator path on
  OVPhysX at large environment counts.
