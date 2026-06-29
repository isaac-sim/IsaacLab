Changed
^^^^^^^

* Changed the Cartpole task lighting from :class:`~isaaclab.sim.DomeLightCfg` to
  :class:`~isaaclab.sim.DistantLightCfg` angled at 20° roll and 20° pitch. Unified light
  intensity (2000) and color (0.75, 0.75, 0.75) across the direct, camera, and manager-based
  variants via shared constants in ``constants.py``. Updated lighting-dependent test golden images
  accordingly.
