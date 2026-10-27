* Fixed :func:`~isaaclab.sim.utils.resolve_matching_prims_from_source` traversing the whole stage when it
  resolves an expression without a clone plan, although it only reads the source instance. A concrete path is now
  looked up directly, and an expression walks only the subtree of its source instance.
