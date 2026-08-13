Fixed
^^^^^

* Fixed URDF conversion producing an asset with no geometry when the URDF referenced its meshes
  through ``package://`` URLs and fixed joints were merged. The ROS package is now derived from the
  URDF's own location, so ``UrdfConverterCfg.ros_package_paths`` only has to be set for packages
  laid out unconventionally.
