Fixed
^^^^^

* Fixed the Newton cloner discarding USD-authored ``physics:approximation`` on
  collision meshes: cloned environments flattened every collision mesh to a single
  convex hull regardless of the authored mode (``convexDecomposition``,
  ``boundingSphere``, ``boundingCube``, ``meshSimplification``, or ``none``),
  without a warning. The cloner now honors authored approximations the same way
  non-cloned scene loading does, and applies the default convex-hull simplification
  only to meshes with no authored approximation. Also switched the core install to
  ``newton[sim,importers]`` so the mesh-processing dependencies (``coacd``,
  ``fast-simplification``, ...) ship by default and ``convexDecomposition``
  decomposes instead of silently falling back to a single convex hull.
