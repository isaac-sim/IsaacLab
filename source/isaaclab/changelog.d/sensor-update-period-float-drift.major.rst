Changed
^^^^^^^

* **Breaking:** Added a float64 ``_elapsed_since_update`` buffer to the timing state of
  :class:`~isaaclab.sensors.SensorBase` and passed it to the ``update_timestamp_kernel``,
  ``update_outdated_envs_kernel`` and ``reset_envs_kernel`` kernels in place of the float32 timestamps
  they used to subtract. The timing buffers are now allocated by ``SensorBase._create_timing_buffers()``.
  Sensors that allocate or resize these private buffers themselves, for example after changing
  ``_num_envs``, must call ``_create_timing_buffers()`` instead.

Fixed
^^^^^

* Fixed :class:`~isaaclab.sensors.SensorBase` refreshing sensors later than their ``update_period`` once the
  sensor clock had run for some seconds. Whether a sensor is due was decided by subtracting two float32
  timestamps that grow with the episode, and their rounding error exceeded the ``1e-6`` tolerance, so sensors
  with ``update_period`` set to a multiple of the physics time step skipped refreshes (for example, the height
  scanner of the locomotion tasks from 16 s into an episode, and sensors with long update periods at small
  time steps within the first second). The time since the last refresh is now accumulated in float64.
