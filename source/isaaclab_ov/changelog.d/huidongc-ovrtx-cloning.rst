Fixed
^^^^^

* Fixed cloned environments disappearing from tiled camera output if
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.use_cloning` is set to ``True``,
  by correcting scene-partition attribute creation on env roots and cameras.

Changed
^^^^^^^

* Changed the default of :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.use_cloning` to ``True``. This will bring
  notable speedup for the total startup time (Launch to Train), esp. for large-scale env setups. On
  Isaac-Dexsuite-Kuka-Allegro-Lift-v0 with 1024 env clones, the total startup time (Launch to Train) dropped from
  ~78s to ~43s.
