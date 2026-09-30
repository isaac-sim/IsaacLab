* Changed masked :meth:`~isaaclab.sensors.ray_caster.BaseRayCaster.reset` to resample drift in a Warp kernel from
  per-environment random states seeded from torch's global generator, so it neither synchronizes the host nor
  allocates and can be captured in a CUDA graph. When both ``env_ids`` and ``env_mask`` are given, the mask selects
  the resampled environments, matching :meth:`~isaaclab.sensors.SensorBase.reset`.
* Changed :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.reset` to reuse preallocated per-DOF masks, so
  masked resets of Newton-native actuator state do not allocate and can be captured in a CUDA graph.
