* Fixed Isaac Sim PhysX 6.2 contact sensors to report total contact forces as normal plus friction,
  including filtered forces and histories, and to report aggregate friction without filters or a
  detailed-contact capacity. Isaac Sim PhysX 6.1 retains its normal-only compatibility behavior.

* Fixed ``friction_forces_w`` on Isaac Sim PhysX 6.2 to return the documented aggregate vector
  shape instead of the filtered friction matrix shape.
