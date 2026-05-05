Removed
^^^^^^^

* Removed ``ArticulationData.body_incoming_joint_wrench_b`` to match the
  shared articulation data API. Code that needs incoming joint reaction
  wrenches should use a backend joint-wrench sensor instead of the articulation
  data object.
