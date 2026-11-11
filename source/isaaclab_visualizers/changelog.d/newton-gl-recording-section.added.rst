* Added a Recording section to the Newton GL HUD that records the window to an ``mp4`` video and the policy being played
  or trained to a ``.pt`` checkpoint. A video stays in a temporary file when stopped and is kept with Save.
* Added :class:`~isaaclab_visualizers.newton.NewtonRecordingCfg` (``NewtonGLBackendCfg.recording``) with the
  output folder, file prefix, auto-numbering, and timestamp options shared by all recordings, plus the video frame
  rate and a video length after which recording stops by itself.
