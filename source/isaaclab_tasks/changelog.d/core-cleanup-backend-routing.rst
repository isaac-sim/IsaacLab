Fixed
^^^^^

* Removed the redundant backend-specific Jacobian refresh from the NIST
  ``reset_end_effector_around_asset`` event. Articulation data refreshes forward kinematics on demand
  after joint writes, avoiding access to ``root_physx_view`` on OVPhysX.
