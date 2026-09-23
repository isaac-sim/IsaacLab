Fixed
^^^^^

* Fixed :func:`~isaaclab.sensors.camera.utils.create_pointcloud_from_rgbd` raising a ``TypeError`` when
  ``rgb`` is a color tuple or ``None``.
* Fixed :func:`~isaaclab.utils.math.quat_slerp` negating the caller's ``q2`` tensor in place when
  taking the shorter arc.
* Fixed :func:`~isaaclab.utils.sensors.convert_camera_intrinsics_to_usd` not warning about aperture
  offsets when the principal point is left of or above the image center.
* Fixed :meth:`~isaaclab.utils.datasets.HDF5DatasetFileHandler.create` failing when ``file_path`` is a
  bare file name without a directory.
