Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.schemas.apply_rigid_body_properties`,
  :func:`~isaaclab.sim.schemas.apply_collision_properties`, and
  :func:`~isaaclab.sim.schemas.apply_mass_properties` force-applying their defining USD API on
  the input prim. They now modify the prims in the subtree that already carry the API, matching
  the legacy nested writers, and only apply a fresh API on the input prim when the subtree
  carries none. Previously, passing a fragment list for ``rigid_props`` on a USD asset whose
  spawn prim carries the articulation root turned the asset's links into nested rigid bodies,
  and the PhysX parser dropped the articulation's joints.
* Fixed :func:`~isaaclab.sim.schemas.apply_collision_properties` unconditionally returning
  ``True``. It now raises ``ValueError`` on an invalid prim path and reports fragment failures
  and skipped instanced prims in its return value, matching
  :func:`~isaaclab.sim.schemas.apply_rigid_body_properties` and
  :func:`~isaaclab.sim.schemas.apply_mass_properties`.
