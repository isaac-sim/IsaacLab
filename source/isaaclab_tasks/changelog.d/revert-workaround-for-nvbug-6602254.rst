Removed
^^^^^^^

* Removed the ``partition_bounds_marker_min`` and ``partition_bounds_marker_max`` scene
  entries from ``Isaac-Lift-Cable-Franka`` and ``Isaac-Lift-Cable-Franka-Camera``. Those
  millimetre-scale cubes pinned each Isaac RTX scene partition to the workspace as a
  workaround for Kit RTX not refreshing animated ``UsdGeom.BasisCurves`` bounding boxes
  (OMPE-105749 / NVBug 6602254). Custom environments that copied the markers can drop
  them; ``test_rendering_franka_cable_partition_bounds`` remains as the regression guard.
