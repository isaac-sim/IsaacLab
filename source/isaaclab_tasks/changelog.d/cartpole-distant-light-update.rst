Changed
^^^^^^^

* Changed the Cartpole task lighting from :class:`~isaaclab.sim.DomeLightCfg` to
  :class:`~isaaclab.sim.DistantLightCfg` angled at 20° roll and 20° pitch. Unified light
  intensity and color across the direct, camera, and manager-based variants via
  ``CARTPOLE_DISTANT_LIGHT_INTENSITY``, ``CARTPOLE_DISTANT_LIGHT_COLOR``, and
  ``CARTPOLE_DISTANT_LIGHT_ORIENTATION`` in ``constants.py``. Updated lighting-dependent test golden images
  accordingly.
