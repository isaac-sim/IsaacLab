* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.max_envs_per_render_product` to split the camera sensors
  of an OVRTX renderer over several render products. Each render product renders a contiguous range of
  environments into its own, smaller tiled image, so a very large camera sensor can render. An integer sets the
  maximum number of environments in one render product, and ``"all"`` keeps one render product. Added a note that
  names this setting to the error about a failed frame.
* Added a split search to the OVRTX renderer for the default ``"auto"``. If the first frame fails, the renderer
  halves the environments per render product of one camera sensor and renders the frame again. One frame tries at
  most four smaller layouts. If a smaller layout renders, a warning gives the integer that skips the search on
  later runs. The new render products get the attributes of the failed render product, so render settings that a
  script wrote onto it stay in effect. With asynchronous rendering, a step waits for its frame until every camera
  sensor on the engine has a rendered frame.
