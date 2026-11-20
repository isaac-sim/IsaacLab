* Changed the default of :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product` from ``"all"`` to
  ``"auto"``, so a camera sensor that fails its first frame with one render product can still render with the
  OVRTX renderer. Set ``"all"`` to keep one render product in every case.
