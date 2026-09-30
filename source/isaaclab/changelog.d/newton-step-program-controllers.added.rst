* Added :attr:`~isaaclab.managers.ActionTerm.apply_every_physics_step` and
  :attr:`~isaaclab.managers.ActionManager.apply_every_physics_step`. Terms whose applied command does not depend on the
  asset's current state (joint position, velocity, and effort targets, binary grippers, tendon targets) set it to
  ``False``.
* Added a ``fold`` argument to :meth:`~isaaclab.physics.PhysicsManager.set_decimation` so an environment can allow or
  forbid folding the decimation loop into one physics step.
