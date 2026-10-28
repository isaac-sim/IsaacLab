* Added a Recording section to the Newton GL HUD that records the window to an ``mp4`` video, the visible
  environments to a ``.viser`` file, and the policy being played or trained to a ``.pt`` checkpoint. A video
  stays in a temporary file when stopped and is kept with Save.
* Added Visible Envs support to the OVRTX view of the Newton GL window.
* Added :class:`~isaaclab_visualizers.newton.NewtonRecordingCfg` (``NewtonGLBackendCfg.recording``) with the
  output folder, file prefix, auto-numbering, and timestamp options shared by all recordings, plus the video frame
  rate and a video length after which recording stops by itself.
* Added animation playback to the Newton visualizer's Recording section: "Open..." lists ``.viser`` recordings, and the
  loaded animation is drawn as a tinted ghost over the simulation. A transport bar along the bottom of the window has
  its own Play/Pause, a draggable timeline, Loop, and Speed, independent of the simulation's pause. The Recording
  section keeps "Hide live simulation" and Show markers, Opacity, Tint, Offset, and Frame options. Animations record
  the visualization markers, such as velocity arrows, and replay them with their changing scales.
* Recording a Policy Ghosts session now writes a ``.json`` file next to the ``.viser`` animation that names each slot
  and its tint, and places the ghost slots on the reference as in the window. Opening the animation lists the slots with
  a checkbox each, tints and fades the ghosts, and hides the live simulation, so a ghost comparison can be scrubbed
  frame by frame.
