Fixed
^^^^^

* Fixed the NIST ``reset_end_effector_around_asset`` event accessing ``root_physx_view`` on OVPhysX, whose
  articulation has no such view. It now refreshes the PhysX Jacobians only when the robot exposes that view.
