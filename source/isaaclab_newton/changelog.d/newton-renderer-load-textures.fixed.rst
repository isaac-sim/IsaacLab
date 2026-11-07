* Fixed :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` loading mesh textures at startup when
  :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.enable_textures` is ``False``; a texture Newton cannot
  load, such as a UDIM texture, no longer fails a renderer that does not use textures.
