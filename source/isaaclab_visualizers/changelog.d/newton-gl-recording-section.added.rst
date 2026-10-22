* Added a Recording section to the Newton GL HUD that records the window to an ``mp4`` video, the visible
  environments to a ``.viser`` file, and the simulation state to an ``npz`` checkpoint.
* Added Visible Envs support to the OVRTX view of the Newton GL window.
* Added :class:`~isaaclab_visualizers.newton.NewtonRecordingCfg` (``NewtonGLBackendCfg.recording``) with the
  output folder, file prefix, auto-numbering, and timestamp options shared by all recordings, plus the video frame
  rate and a video length after which recording stops by itself.
