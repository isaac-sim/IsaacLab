Changed
^^^^^^^

* Acquired Newton viewer models from the simulation's shared backend registry instead of the
  physics manager. Viewers requested transforms and geometry directly through SDP; headless GL
  and RTX captures requested current arrays only when a frame was requested.
* Rebound GL, RTX, Rerun, and Viser resources after hard resets, including when picking was disabled.
* Shared declarative ``newton_cfg`` inputs with streaming-camera renderers; initialization acquired
  the clone-built model through ``get_or_create_backend(cfg)`` without cfg notifications.
