* Added ``env_index``, ``track_path``, ``follow_heading``, and ``heading_smoothing_time_constant`` to
  :class:`~isaaclab.visualizers.PerspectiveCameraCfg`, and :class:`~isaaclab.visualizers.camera_follower.CameraFollower`,
  so a perspective view can frame an environment or follow an asset with smoothed yaw. Only the Newton GL window
  renders these views; other visualizers ignore them.
