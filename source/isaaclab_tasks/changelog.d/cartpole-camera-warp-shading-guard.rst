Fixed
^^^^^

* Fixed the cartpole camera tasks (``Isaac-Cartpole-Camera`` and ``Isaac-Cartpole-Camera-Direct``)
  accepting ``presets=newton_renderer`` together with ``simple_shading_constant_diffuse``,
  ``simple_shading_diffuse_mdl``, or ``simple_shading_full_mdl``. The Newton Warp renderer does not
  publish those data types, so the run failed only at environment construction, after the simulator
  had started. The configurations now reject the combination during config resolution, mirroring the
  guard already used by the Shadow Hand camera task. Use a data type the Warp renderer supports, for
  example ``presets=newton_renderer,rgb``, or keep the shading data types on an RTX backend with
  ``presets=isaacsim_rtx,simple_shading_full_mdl``.
