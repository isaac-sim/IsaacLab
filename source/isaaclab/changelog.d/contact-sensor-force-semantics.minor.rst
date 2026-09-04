Added
^^^^^

* Added ``net_forces_w`` as the total contact force (normal + friction), with matching history
  and filtered-matrix properties. PhysX and OVPhysX cannot compute a total force, so they
  return the corresponding normal-force quantity and warn.
* Added ``friction_forces_w`` as the aggregate friction force. PhysX and OVPhysX only provide
  filtered friction, so they return ``friction_force_matrix_w`` and warn.
* Added explicit aggregate and filtered normal and friction force contracts, including
  ``net_friction_forces_w_history`` and ``friction_force_matrix_w_history``.
