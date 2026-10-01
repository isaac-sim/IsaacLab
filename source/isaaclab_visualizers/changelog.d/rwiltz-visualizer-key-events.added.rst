* Added ``key_event_source`` to :class:`~isaaclab_visualizers.kit.KitVisualizer`, which reports
  physical keys, and :class:`~isaaclab_visualizers.newton.NewtonGLVisualizer`, which reports
  layout-mapped keys, withholds keys typed into its UI, and suspends its WASD/QE and arrow-key
  camera movement while the keyboard is captured. Headless runs and the Newton RTX, Viser and
  Rerun visualizers return ``None``.
