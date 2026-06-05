Added
^^^^^

* Added :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg`, the ``physxArticulation:*``
  single-namespace articulation-root fragment (PhysX ``PhysxArticulationAPI``). It carries
  ``articulation_enabled``, ``enabled_self_collisions``, solver position / velocity iteration
  counts, and sleep / stabilization thresholds, and composes in an ``articulation_props`` fragment
  list applied via :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`.
