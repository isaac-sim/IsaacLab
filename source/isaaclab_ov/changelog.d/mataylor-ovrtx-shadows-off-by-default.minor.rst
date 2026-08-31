Added
^^^^^

* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.enable_shadows`, which authors
  ``omni:rtx:minimal:castShadows`` on the OVRTX render product. It applies to the
  ``simple_shading_*`` data types, which are the ones that select RTX Minimal mode; OVRTX's
  path-traced modes provide no shadow switch and always cast shadows.

Changed
^^^^^^^

* Changed the OVRTX renderer to turn shadows off by default in RTX Minimal mode. Renders that need
  cast shadows from the ``simple_shading_*`` data types must now set
  ``OVRTXRendererCfg(enable_shadows=True)``.
