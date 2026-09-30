* Changed masked :meth:`~isaaclab.sensors.ray_caster.BaseRayCaster.reset` to resample drift without a host
  synchronization. When both ``env_ids`` and ``env_mask`` are given, the mask selects the resampled environments,
  matching :meth:`~isaaclab.sensors.SensorBase.reset`.
