Changed
^^^^^^^

* Extracted the repeated solver-kwargs filtering pattern from
  :class:`~isaaclab_newton.physics.NewtonFeatherstoneManager`,
  :class:`~isaaclab_newton.physics.NewtonMJWarpManager`, and
  :class:`~isaaclab_newton.physics.NewtonXPBDManager` into a shared
  :meth:`~isaaclab_newton.physics.NewtonManager._filter_solver_kwargs` helper,
  so :class:`NewtonManager` subclasses can reuse it when forwarding
  ``solver_cfg`` fields to a Newton solver constructor.
