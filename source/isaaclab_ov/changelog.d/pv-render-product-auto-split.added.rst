* Added a split search to the OVRTX renderer. If the first frame fails, the renderer halves the environments per
  render product of a camera sensor with ``"auto"`` :attr:`~isaaclab.sensors.CameraCfg.max_envs_per_render_product`
  and renders the frame again. A warning gives the integer that skips this search. The new render products
  get the attributes of the failed render product, so render settings that a script wrote onto it stay in effect.
  With asynchronous rendering, a step waits for its frame while a camera sensor on the engine has no rendered
  frame yet.
