Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.assets.RigidObjectCollection` selecting rigid bodies outside the
  collection whose names share the members' prefix or suffix, and rejecting members at different
  path depths. The collection view now matches each configured prim path exactly.
