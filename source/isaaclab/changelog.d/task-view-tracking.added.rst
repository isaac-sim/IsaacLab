* Added ``env_index``, ``track_path``, ``follow_heading``, and ``heading_smoothing_time_constant`` to
  :class:`~isaaclab.visualizers.PerspectiveCameraCfg`, so a perspective view can frame an environment or follow an asset
  with smoothed yaw. Added the ``viz:<type>:tracking`` video source, which records the first such view whichever view
  the window shows, for Newton GL.
* Added :meth:`~isaaclab.visualizers.camera_follower.CameraFollower.set_track_path`, ``asset_names``, and
  ``body_names`` so a view can switch the asset or body it follows while running.
