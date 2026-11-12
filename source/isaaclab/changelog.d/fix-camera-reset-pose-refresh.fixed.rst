* Fixed :class:`~isaaclab.sensors.Camera` with :attr:`~isaaclab.sensors.CameraCfg.update_latest_camera_pose`
  disabled keeping the previous episode's pose after a reset. The reset read the pose before reset events moved
  the camera; the pose is now read at the first capture after the reset, unless
  :meth:`~isaaclab.sensors.Camera.set_world_poses` or :meth:`~isaaclab.sensors.Camera.set_world_poses_from_view`
  set it after the reset. Corrected the flag's documentation, which stated that the pose from initialization is
  returned.
