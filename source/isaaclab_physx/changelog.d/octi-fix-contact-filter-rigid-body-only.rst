Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.sensors.ContactSensor` incorrectly matching geometry children
  when resolving ``filter_prim_paths_expr``. URDF-to-USD conversion creates prim hierarchies
  where a rigid body and its geometry child share the same leaf name (e.g. ``Object`` and
  ``Object/Object``). PhysX's glob engine prefix-matches bare leaf patterns, finding both
  prims and raising a count-consistency error. The fix wraps the leaf segment in PhysX
  alternation syntax ``(leaf)`` before the ``.*`` → ``*`` conversion, anchoring the match
  to the correct path depth.
