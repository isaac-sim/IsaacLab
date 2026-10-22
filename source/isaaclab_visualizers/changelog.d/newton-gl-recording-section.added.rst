* Added a Recording section to the Newton GL HUD that records the window to an ``mp4`` video, the visible
  environments to a ``.viser`` file, and the policy being played or trained to a ``.pt`` checkpoint. A video
  stays in a temporary file when stopped and is kept with Save.
* Added Visible Envs support to the OVRTX view of the Newton GL window.
* Added :class:`~isaaclab_visualizers.newton.NewtonRecordingCfg` (``NewtonGLBackendCfg.recording``) with the
  output folder, file prefix, auto-numbering, and timestamp options shared by all recordings, plus the video frame
  rate and a video length after which recording stops by itself.
* Added animation playback to the Newton GL Recording section: "Open..." lists ``.viser`` recordings, and a loaded
  animation plays next to the simulation with play, pause, scrub, speed, loop, opacity, and offset controls, and an
  option to hide the simulation. The animation is drawn as separate shapes and does not touch the Newton model.
* Loading an animation in the Newton GL window enters a Replay mode: the simulation is hidden and paused, the
  Playback section shows the replay transport (play, step, loop, scrub, speed) with an Exit Replay button, a header
  badge names the clip, and the camera frames it. The View choice switches between Replay only, Overlay on live, and
  Side by side, and leaving the mode restores the previous pause and visibility state.
