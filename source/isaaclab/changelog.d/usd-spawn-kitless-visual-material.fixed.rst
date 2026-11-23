* Fixed :class:`~isaaclab.sim.UsdFileCfg` skipping its ``visual_material`` when Kit is not running, which left
  visualization markers in kitless renderers such as OVRTX with the asset's own color instead of the configured one.
