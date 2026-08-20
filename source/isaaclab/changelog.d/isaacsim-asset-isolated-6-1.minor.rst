Fixed
^^^^^

* Fixed URDF conversion dropping every ``package://`` mesh when ``merge_fixed_joints`` is enabled,
  which produced an asset with the full joint and rigid-body hierarchy but no geometry. Converting
  a ROS-style description no longer requires setting ``ros_package_paths`` by hand.

Changed
^^^^^^^

* Changed the pinned standalone URDF/MJCF importers from ``isaacsim-asset-isolated`` 6.0 to 6.1.
