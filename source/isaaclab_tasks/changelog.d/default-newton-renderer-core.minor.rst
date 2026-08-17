Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_tasks.utils.presets.MultiBackendRendererCfg` to use the Newton renderer
  by default. Select ``renderer=isaacsim_rtx`` to continue using the Isaac RTX renderer.
  Contributed teleoperation and Galbot Stack cameras retain their explicit Isaac RTX defaults.
