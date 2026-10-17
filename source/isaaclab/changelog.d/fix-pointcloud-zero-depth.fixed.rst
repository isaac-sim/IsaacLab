* Fixed :func:`~isaaclab.sensors.camera.utils.create_pointcloud_from_depth` and
  :func:`~isaaclab.sensors.camera.utils.create_pointcloud_from_rgbd` keeping pixels with zero depth as points at
  the camera position when invalid points are removed. Zero depth now counts as invalid, as documented, and the
  colors from :func:`~isaaclab.sensors.camera.utils.create_pointcloud_from_rgbd` stay aligned with the kept points.
