* Replaced ``OVRTXRendererCfg.max_envs_per_render_product`` by the environment variable
  ``ISAAC_LAB_OVRTX_MAX_ENVS_PER_RENDER_PRODUCT``, because the render product split is a temporary workaround for a
  renderer limit and not a feature. The attribute was never in a release. Unset or ``auto`` keeps the search after a
  failed first frame, ``all`` keeps one render product, and an integer sets the maximum number of environments in one
  render product.
