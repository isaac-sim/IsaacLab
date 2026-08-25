Fixed
^^^^^

* Fixed the cable disappearing from camera images in ``Isaac-Lift-Cable-Franka`` and
  ``Isaac-Lift-Cable-Franka-Camera`` when Isaac RTX per-environment scene partitioning is enabled.
  Kit RTX never refreshes the bounding box of an animated ``UsdGeom.BasisCurves`` prim (OMPE-105749),
  so the partition was sized from the cable's spawn extent and culled the cable once it deformed
  outside it -- measured on Kit 110.1.2, the cable vanished at 0.6 m of displacement. Added the
  ``partition_bounds_marker_min`` and ``partition_bounds_marker_max`` scene entries, two 1 mm static
  visual cubes at diagonally opposite corners of the workspace, which pin the partition bounds to the
  full workspace volume; cable visibility then matches an unpartitioned render exactly. The markers
  carry no colliders and do not participate in physics, and they are set to ``None`` when scene
  partitioning is disabled. Remove them once Kit updates animated-curve bounding boxes.

Added
^^^^^

* Added ``test_rendering_franka_cable_partition_visibility``, which moves the cable 0.7 m past its
  spawn extent under Isaac RTX scene partitioning and asserts it still renders. The existing AOV
  tests capture a settled cable, so they never leave the bounding box Kit RTX computes at spawn.
