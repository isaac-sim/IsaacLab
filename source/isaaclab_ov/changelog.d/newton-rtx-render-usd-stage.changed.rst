* Moved the clone-plan copy enumeration that the OVRTX renderer's ``_clone_sources`` ran into
  :func:`~isaaclab_ov.renderers.ovrtx_usd.iter_clone_copies`, so the renderer and
  :func:`~isaaclab_ov.stage.create_render_ovstage` clone the same set of prims. Cloning behavior is unchanged.
