Fixed
^^^^^

* Fixed cloned environments disappearing from tiled camera output if
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.use_cloning` is set to ``True``,
  by correcting scene-partition attribute creation on env roots and cameras.

Changed
^^^^^^^

* Changed the default of :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.use_cloning` to ``True``.
