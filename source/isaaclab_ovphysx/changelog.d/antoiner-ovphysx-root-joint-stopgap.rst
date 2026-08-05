Fixed
^^^^^

* Fixed OvPhysX failing to build any articulation for assets whose root joint targets its own
  ``UsdPhysics.ArticulationRootAPI`` prim as ``body0`` instead of leaving ``body0`` empty, which
  previously surfaced as ``could not create any articulation bindings``. Such joints are now
  normalized to the empty-``body0`` world-attachment spelling before the stage is handed to the
  OvPhysX parser.
