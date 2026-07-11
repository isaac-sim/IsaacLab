Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.NewtonArticulationCfg`, the ``newton:*``
  single-namespace articulation-root fragment (``newton:selfCollisionEnabled`` via
  ``NewtonArticulationRootAPI``). It composes with
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg` in an ``articulation_props`` fragment
  list applied via :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`.
