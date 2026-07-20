Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.schemas.apply_rigid_body_properties` and
  :func:`~isaaclab.sim.schemas.apply_mass_properties` stopping at the outermost schema-bearing
  prim on each branch. On assets with nested rigid-body hierarchies (child links authored under
  their parent link prims, as produced by the URDF importer in Isaac Sim 6.0 and later), only
  the outermost link received the fragment properties. The writers now modify every carrier in
  the subtree, matching the legacy writers' full-subtree traversal.
