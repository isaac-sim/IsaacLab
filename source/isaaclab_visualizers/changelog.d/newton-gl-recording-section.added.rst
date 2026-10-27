* Added a Recording section to the Newton GL HUD that records the window to an ``mp4`` video, the visible
  environments to a ``.viser`` file, and the policy being played or trained to a ``.pt`` checkpoint. A video
  stays in a temporary file when stopped and is kept with Save.
* Added Visible Envs support to the OVRTX view of the Newton GL window.
* Added :class:`~isaaclab_visualizers.newton.NewtonRecordingCfg` (``NewtonGLBackendCfg.recording``) with the
  output folder, file prefix, auto-numbering, and timestamp options shared by all recordings, plus the video frame
  rate and a video length after which recording stops by itself.
* Added animation ghosts to the Newton visualizer's Policy Ghosts section: "Add Animation..." lists ``.viser``
  recordings, and each one opens as a tinted ghost with its own opacity, tint, and offset next to the simulation.
  Several animations can be open at once; they need no extra simulated environments and work without
  ``--policy_ghosts``. They share one timeline with loop and speed controls, play and pause with the Playback
  section's Pause Simulation button, and the timeline can be dragged when only animations are loaded. "Hide live
  simulation" shows only the animations. Animations record the visualization markers, such as velocity arrows, and
  replay them with their changing scales.
