* Fixed OVPhysX contact sensors to report total contact forces as normal plus friction,
  including filtered forces and histories. Added aggregate friction reporting without
  contact filters and removed the detailed-contact capacity requirement for friction forces.
  Requires OVPhysX 0.6.4.
* Fixed ``friction_forces_w`` to return the documented aggregate vectors with shape
  ``(E, S, 3)``. Use ``friction_force_matrix_w`` for filtered forces with shape ``(E, S, F, 3)``.
