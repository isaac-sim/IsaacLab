Fixed
^^^^^

* Fixed the cable disappearing from tiled camera images in ``Isaac-Lift-Cable-Franka`` and
  ``Isaac-Lift-Cable-Franka-Camera`` when Isaac RTX per-environment scene partitioning is enabled.
  Kit RTX never refreshes the bounding box of an animated ``UsdGeom.BasisCurves`` prim (OMPE-105749),
  so the partition was sized from the cable's initial extent and culled the cable once it moved
  outside it. Added the ``partition_bounds_marker_min`` and ``partition_bounds_marker_max`` scene
  entries -- two 1 mm static visual cubes at diagonally opposite corners of the workspace -- to pin
  the partition bounds to the full workspace volume. The markers carry no colliders and do not
  participate in physics, and they are set to ``None`` when scene partitioning is disabled. Remove
  them once Kit updates animated-curve bounding boxes.
