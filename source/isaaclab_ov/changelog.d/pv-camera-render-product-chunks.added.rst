* Added render product splits to the OVRTX renderer. With an integer
  :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product`, each render product of a camera sensor
  renders a contiguous range of environments into its own tiled atlas. With ``"auto"``, the renderer halves
  the environments per render product after a failed first frame and logs the value that skips this search.
