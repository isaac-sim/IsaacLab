* Changed :func:`~isaaclab_ov.renderers.ovrtx_shader_cache.redirect_shader_cache` to take no
  renderer configuration. It now only loads the ovrtx library to apply the cache path, leaving
  initialization and its configuration to the first renderer. Its helper
  ``apply_shader_cache_settings`` is removed; call ``redirect_shader_cache`` instead.
