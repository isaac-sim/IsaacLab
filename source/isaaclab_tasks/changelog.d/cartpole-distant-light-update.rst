Changed
^^^^^^^

* Changed the Cartpole task lighting from :class:`~isaaclab.sim.DomeLightCfg` to
  :class:`~isaaclab.sim.DistantLightCfg` with unified light
  intensity (``CARTPOLE_DISTANT_LIGHT_INTENSITY = 2000.0``), color
  (``CARTPOLE_DISTANT_LIGHT_COLOR = (1.0, 1.0, 1.0)``), and orientation
  (``CARTPOLE_DISTANT_LIGHT_ORIENTATION``, -45° pitch and -45° yaw) across the direct, camera,
  and manager-based variants via shared constants in ``constants.py``. Updated lighting-dependent
  test golden images accordingly.
