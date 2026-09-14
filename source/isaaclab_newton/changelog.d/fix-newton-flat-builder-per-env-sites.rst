Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager._cl_inject_sites_fallback` (the
  non-replicated/flat-builder path, used when ``replicate_physics=False``) collapsing every
  environment's matched site index into a single one-element outer list instead of one sublist
  per environment. This raised ``IndexError`` in any consumer indexing ``per_world[env_index]``
  for ``env_index > 0`` (e.g. :meth:`~isaaclab_newton.sensors.ray_caster.NewtonRaycastSensor._resolve_site_indices`)
  as soon as a scene had more than one environment with ``replicate_physics=False``.
