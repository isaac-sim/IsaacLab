* Added ``key_event_source`` to :class:`~isaaclab_visualizers.kit.KitVisualizer`, which reports
  physical keys, and to :class:`~isaaclab_visualizers.newton.NewtonGLVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizer`, which report layout-mapped keys and
  withhold presses typed into their UI. Headless runs and the Viser and Rerun visualizers return
  ``None``.
* Added ``set_camera_keys_enabled`` to the Newton GL and RTX visualizers, which turns their WASD/QE
  and arrow-key camera movement off and on.
