* Fixed graph-replay invalidation from before a hard ``sim.reset()`` carrying over to the rebuilt model, which
  could make asset pose reads in ``PHYSICS_READY`` callbacks raise a ``RuntimeError`` before the solver is rebuilt.
