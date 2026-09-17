Fixed
^^^^^

* Restored hand-capsule and finger-pad contacts for rigid Franka Lift and Reorient by
  disabling the primitive asset's arm capsules through a task-local spawn override.
  This replaced the convex-hull selection, which also enabled arm and hand mesh contacts.
* Made Lift/Reorient reset clearance ignore colliders with disabled collision, preventing
  inactive arm capsules from blocking reset sampling. Reset acceptance now follows enabled
  collision geometry; disabled arm meshes are no longer used as clearance proxies.
