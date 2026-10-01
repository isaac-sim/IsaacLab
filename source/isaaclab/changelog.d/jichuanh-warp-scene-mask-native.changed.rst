* Changed masked :meth:`~isaaclab.sensors.ray_caster.BaseRayCaster.reset` to resample drift in a Warp kernel from
  per-environment random states seeded from torch's global generator, so it neither synchronizes the host nor
  allocates and can be captured in a CUDA graph. When both ``env_ids`` and ``env_mask`` are given, the mask selects
  the resampled environments, matching :meth:`~isaaclab.sensors.SensorBase.reset`.
* Changed :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.reset` to reuse preallocated per-DOF masks, so
  masked resets of Newton-native actuator state do not allocate and can be captured in a CUDA graph.
* Changed :meth:`~isaaclab.actuators.ActuatorBase.reset` to also receive a 1-D boolean mask of the environments to
  reset, which masked resets pass in place of indices. Custom actuator models must accept a mask, for example by
  resetting their buffers through :func:`~isaaclab.utils.array.index_fill_`.
* Changed :class:`~isaaclab.actuators.DelayedPDActuator` to raise :class:`ValueError` at construction unless
  ``0 <= min_delay <= max_delay``, instead of at a reset, because masked resets apply the sampled delays without
  checking them on the host.
* Changed the ``limit`` annotation of
  :meth:`~isaaclab.assets.BaseArticulation.set_fixed_tendon_position_limit_index` and
  :meth:`~isaaclab.assets.BaseArticulation.set_fixed_tendon_position_limit_mask` to exclude ``float``, matching the
  backends, which raise :class:`ValueError` for a float because it cannot hold both bounds.
