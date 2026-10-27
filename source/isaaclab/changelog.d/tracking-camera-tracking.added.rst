* Added :class:`~isaaclab.visualizers.TrackingCameraCfg`, a perspective camera that frames an environment or follows
  an asset with optional smoothed heading, and :class:`~isaaclab.visualizers.camera_follower.CameraFollower`, which
  can switch the followed asset while running. Added the ``viz:<type>:tracking`` video source, which records the first
  tracking camera whichever view the window shows, for Newton GL.
