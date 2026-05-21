Fixed
^^^^^

* Excluded ``pytetwild`` install on aarch64 platforms. The package has no aarch64 wheel on PyPI and its
  source build fails (the ``geogram`` CMake dep hardcodes ``-m64``). The single call site in
  :mod:`isaaclab.sim.schemas` already raises a clear "install pytetwild manually or provide a
  pre-tetrahedralized UsdGeom.TetMesh" message when the lazy import fails, so aarch64 users keep
  everything except automatic volume-deformable tetrahedralization.
