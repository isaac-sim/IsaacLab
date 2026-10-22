* Changed the Newton GL and Newton RTX HUD layout: the physics backend, up axis, and gravity now sit in a compact
  strip above the sections, the former "Simulation" section is split into "Playback" and "Environments", and
  "Rendering Options" starts collapsed. The Visible Envs slider moved from the playback controls to "Environments".
  Callbacks registered with ``register_ui_callback(position="side")`` now appear under "Playback".
* Added an always-visible header to the Newton GL HUD showing the task and run, physics backend and device,
  step and simulation time, simulation rate and real-time factor, and status badges
  for replay, pause, and Policy Ghosts. "Environments", "Visualization Markers", and "Model Overlays" are now
  grouped in one "Scene" section.
* Renamed the Newton GL window and its HUD panel from "Newton Viewer" to "Newton Visualizer", and moved
  "Visualization Markers" into a section of its own that is always shown, with a note when a task registers none.
* Moved the recording status text out of the HUD header into the Recording section: a red REC line with the elapsed
  time, an unsaved-video note, and a green confirmation after saving. The section title turns red while recording.
* Removed the up axis and gravity line from the Newton GL HUD header and added tooltips to the header lines, the
  model overlay options, the visualization marker toggles, and the Policy Ghosts slot controls. The default
  recording output folder is now ``recording`` in the working directory, which Git ignores.
* Renamed the Newton GL HUD "View" section to "Camera View", moved "Visualization Markers" back into "Scene"
  (after Visible Envs), and added tooltips to the section headers.
* Renamed Newton's "Selection API" panel to "Articulation Inspector" in the Newton GL HUD.
