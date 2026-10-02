* Changed the Newton GL and Newton RTX HUD layout: the physics backend, up axis, and gravity now sit in a compact
  strip above the sections, the former "Simulation" section is split into "Playback" and "Environments", and
  "Rendering Options" starts collapsed. The Visible Envs slider moved from the playback controls to "Environments".
  Callbacks registered with ``register_ui_callback(position="side")`` now appear under "Playback".
