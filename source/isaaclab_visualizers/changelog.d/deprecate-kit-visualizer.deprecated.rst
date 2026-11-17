* Deprecated :class:`~isaaclab_visualizers.kit.KitVisualizer` and
  :class:`~isaaclab_visualizers.kit.KitBackendCfg` for removal in 3.2. Creating the visualizer, including
  through ``--viz kit`` and ``viz:kit``, emits a :class:`FutureWarning`. Use the Newton GL visualizer
  (``--viz newton_gl``) for the interactive window, tracking cameras and RTX rendering, and the OVRTX
  renderer for Kit-free RTX rendering of sensor cameras. Livestreaming and XR still start Kit and do not
  warn, because they have no replacement yet. Added :attr:`~isaaclab_visualizers.kit.KitBackendCfg.deprecation_warning`
  to turn the warning off for those cases.
