Added
^^^^^

* Added the ``partition_bounds_marker_min`` and ``partition_bounds_marker_max`` scene entries to
  ``Isaac-Lift-Cable-Franka`` and ``Isaac-Lift-Cable-Franka-Camera``: two 1 mm static visual cubes at
  diagonally opposite corners of the workspace. Kit RTX never refreshes the bounding box of an animated
  ``UsdGeom.BasisCurves`` prim (OMPE-105749), so a per-environment scene partition sized from the cable's
  initial extent can cull the cable once it deforms outside that extent. The markers pin the partition
  bounds to the full workspace volume so the cable cannot leave them. They carry no colliders and do not
  participate in physics, and they are set to ``None`` when scene partitioning is disabled. This is a
  preventative workaround: Isaac Lab currently syncs cable points to Hydra through a CPU mirror (NVBug
  6502662), and OMPE-105749 affects the GPU interop path, so the culling is not currently observable in
  these environments. Remove the markers once Kit updates animated-curve bounding boxes.
* Added ``test_rendering_franka_cable_partition_visibility``, which asserts the cable still renders after
  deforming outside its spawn extent under Isaac RTX scene partitioning. The existing AOV tests capture a
  settled cable, so they never leave the bounding box Kit RTX computes for the curve at spawn.
