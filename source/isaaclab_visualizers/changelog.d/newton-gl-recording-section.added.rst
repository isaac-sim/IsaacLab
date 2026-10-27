* Added a Recording section to the Newton GL HUD that records the window to an ``mp4`` video, the visible
  environments to a ``.viser`` file, and the policy being played or trained to a ``.pt`` checkpoint. A video
  stays in a temporary file when stopped and is kept with Save.
* Added Visible Envs support to the OVRTX view of the Newton GL window.
* Added :class:`~isaaclab_visualizers.newton.NewtonRecordingCfg` (``NewtonGLBackendCfg.recording``) with the
  output folder, file prefix, auto-numbering, and timestamp options shared by all recordings, plus the video frame
  rate and a video length after which recording stops by itself.
* Added animation playback to the Newton visualizer's Recording section: "Open..." lists ``.viser`` recordings, and the
  loaded animation is drawn as a tinted ghost over the simulation. It plays and pauses with the Playback section's
  Pause Simulation button and has a draggable timeline, a Loop checkbox, "Hide live simulation", and Speed, Opacity,
  Tint, Offset, and Frame options. The timeline is greyed out while live Policy Ghost checkpoints are loaded, since
  they cannot be scrubbed. Animations record the visualization markers, such as velocity arrows, and replay them with
  their changing scales.
