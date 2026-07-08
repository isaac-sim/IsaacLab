Added
^^^^^

* Added :data:`~isaaclab_ovphysx.tensor_types.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION` alias for
  the per-collision-shape rigid-body material tensor type (static friction, dynamic friction,
  restitution) exposed by the ovphysx wheel. Read and write it through
  :class:`~isaaclab_ovphysx.sim.views.OvPhysxView`, e.g.
  ``root_view.get_attribute(tensor_types.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)``.
