* Added render product splits to the OVRTX renderer. With an integer
  :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product`, each render product of a camera sensor
  renders a contiguous range of environments into its own tiled atlas. ``OVRTXCameraRenderData.products`` then
  has one entry for each render product, and ``render_product_paths`` lists all of their paths. The error about a
  failed frame gets a note that names this setting.
* Added a split search to the OVRTX renderer for camera sensors with the default ``"auto"``. If the first frame
  fails, the renderer halves the environments per render product of one such sensor and renders the frame again.
  One frame tries at most four smaller layouts. If a smaller layout renders, a warning gives the integer that
  skips the search on later runs. The new render products get the attributes of the failed render product, so
  render settings that a script wrote onto it stay in effect. With asynchronous rendering, a step waits for its
  frame until every camera sensor on the engine has a rendered frame.
