Added
^^^^^

* Added :data:`~isaaclab_ovphysx.tensor_types.SHAPE_FRICTION_AND_RESTITUTION` and
  :data:`~isaaclab_ovphysx.tensor_types.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION` aliases for
  the per-collision-shape material tensor types (static friction, dynamic friction,
  restitution) exposed by the ovphysx wheel. Read and write them through
  :class:`~isaaclab_ovphysx.sim.views.OvPhysxView`, e.g.
  ``root_view.get_attribute(tensor_types.SHAPE_FRICTION_AND_RESTITUTION)``.
