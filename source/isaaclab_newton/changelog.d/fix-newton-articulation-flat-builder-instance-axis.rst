Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.assets.ArticulationData._create_simulation_bindings` indexing every
  per-instance sim binding (root pose, joint positions/velocities/limits/targets, body
  mass/inertia/wrench, tendon properties) with a hardcoded ``[:, 0]``, which only holds for the
  replicated builder's ``(num_worlds, 1, ...)`` layout. The flat (non-replicated) builder used when
  ``replicate_physics=False`` produces the opposite layout, ``(1, num_instances, ...)``, so
  ``[:, 0]`` silently collapsed every non-replicated scene with more than one environment down to a
  single instance's data for every binding, including write paths used to drive the robot. The
  per-instance axis is now resolved dynamically from the root transforms' shape.
