Added
^^^^^

* Added :attr:`~isaaclab.renderers.RendererCfg.async_rendering` so pipelined rendering is requested
  the same way for every renderer: ``True`` trades one frame of camera latency for pipelined
  rendering, ``False`` (the default) renders synchronously.
* Added the ``ISAAC_LAB_ASYNC_RENDERING`` environment variable, exposed as
  :data:`~isaaclab.renderers.ASYNC_RENDERING_ENV_VAR`, which overrides
  :attr:`~isaaclab.renderers.RendererCfg.async_rendering` for every renderer. It takes the same
  values, so the asynchronous path can be exercised without naming a camera a task may not define.
* Added :func:`~isaaclab.renderers.resolve_async_rendering_enabled`,
  :func:`~isaaclab.renderers.async_rendering_enabled_from_env`, and
  :func:`~isaaclab.renderers.warn_unsupported_async_rendering` so renderers resolve the setting
  consistently and can report it as unimplemented.
