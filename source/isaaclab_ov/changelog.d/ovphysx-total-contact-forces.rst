Fixed
^^^^^

* Fixed OVPhysX contact sensors to report total contact forces as normal plus friction forces,
  including when separate friction tracking was disabled. Added aggregate friction reporting
  without filters and removed detailed-contact capacity limits from force measurements.
  For normal-only measurements, use ``net_normal_forces_w`` and ``normal_force_matrix_w``.
  For filtered friction, use ``friction_force_matrix_w`` instead of the aggregate alias
  ``friction_forces_w``.
