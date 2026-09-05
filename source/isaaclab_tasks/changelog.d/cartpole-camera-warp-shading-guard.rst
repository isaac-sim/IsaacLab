Fixed
^^^^^

* Fixed ``Isaac-Cartpole-Camera`` and ``Isaac-Cartpole-Camera-Direct`` accepting
  ``presets=newton_renderer`` together with a ``simple_shading_*`` data type, which the Newton Warp
  renderer cannot produce: the run failed only at environment construction, after the simulator had
  started. The combination is now rejected during config resolution. Use ``presets=newton_renderer,rgb``,
  or keep the shading data types on an RTX backend with ``presets=isaacsim_rtx,simple_shading_full_mdl``.
