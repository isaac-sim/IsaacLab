* Fixed pending forward kinematics and graph-replay invalidation from before a hard ``sim.reset()`` carrying over
  to the rebuilt model, which could make transform reads during the reset, such as scene data reads in
  ``PHYSICS_READY`` callbacks, raise a ``RuntimeError``.
