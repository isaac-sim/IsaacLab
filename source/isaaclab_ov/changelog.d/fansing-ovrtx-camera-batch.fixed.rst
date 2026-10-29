* Fixed OVRTX rendering all cameras once per lazily read camera in a step: the renderer declares
  :attr:`~isaaclab.renderers.BaseRenderer.renders_all_cameras`, so cameras read in the same step render together.
