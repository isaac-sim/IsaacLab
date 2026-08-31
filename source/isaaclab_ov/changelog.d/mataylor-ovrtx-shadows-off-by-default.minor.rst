Added
^^^^^

* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.enable_shadows`, which authors
  ``omni:rtx:shadows:enabled`` and ``omni:rtx:minimal:castShadows`` on the OVRTX render product so
  the toggle covers the path-traced and minimal render modes alike.

Changed
^^^^^^^

* Changed the OVRTX renderer to request shadows off by default. Renders that need cast shadows must
  now set ``OVRTXRendererCfg(enable_shadows=True)``.
