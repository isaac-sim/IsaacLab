Fixed
^^^^^

* Fixed cloned environments disappearing from tiled camera output if
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.use_ovrtx_cloning` is set to ``True``,
  by correcting scene-partition attribute creation on env roots and cameras.

Changed
^^^^^^^

* Renamed the ``use_cloning`` field on :class:`~isaaclab_ov.renderers.OVRTXRendererCfg` to ``use_ovrtx_cloning``.
  Changed its default value to ``True``. This will bring notable speedup for the total startup time (Launch to Train),
  esp. for large-scale env setups. On Isaac-Dexsuite-Kuka-Allegro-Lift-v0 with 1024 env clones, the total startup time
  dropped from ~78s to ~43s. Note that if ``use_ovrtx_cloning`` is enabled but the env setup is heterogeneous, the
  OVRTX renderer will disable the internal cloning path and logs a warning, exporting the full multi-environment stage
  instead (same effect as setting ``use_ovrtx_cloning`` to ``False`` for that run).
