Changed
^^^^^^^

* Acquired Newton viewer models from the simulation's shared backend registry instead of the
  physics manager. Viewers requested transforms and geometry directly through SDP; headless GL
  and RTX captures requested current arrays only when a frame was requested.
* Rebound GL, RTX, Rerun, and Viser resources after hard resets, including when picking was disabled.
* Shared the selected Newton representation with streaming-camera renderers; initialization acquired
  the clone-built model through ``get_or_create_backend(cfg)`` without cfg notifications.
* **Breaking:** Replaced streaming renderer nicknames with ``streaming_cam_renderer_cfg``.
  Kit defaulted to ``IsaacRtxRendererCfg()``; Newton, Rerun, and Viser defaulted to
  ``NewtonWarpRendererCfg()``. Explicit renderer failures propagated instead of switching renderer
  or disabling the stream. Visualizers registered their configured auto-camera renderer before cloning.
  Pass a renderer configuration to customize construction.
* Honored explicit auto-camera targets in Rerun and Viser instead of substituting a scene camera.
  Omitting the target continued to adopt an existing camera, as documented.
* Consolidated GL/RTX headless and paused frame handling without changing pause behavior or
  frame readback types.
