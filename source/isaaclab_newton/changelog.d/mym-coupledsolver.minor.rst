Added
^^^^^

* Added ``ke``, ``kd``, and ``mu`` fields to
  :class:`~isaaclab_newton.physics.NewtonShapeCfg`, forwarded onto Newton's
  ``ModelBuilder.default_shape_cfg`` at builder construction. These set the
  per-shape contact defaults for shapes that lack an explicit per-asset
  material; per-asset materials override them. Defaults mirror Newton's
  ``ShapeConfig`` values.
