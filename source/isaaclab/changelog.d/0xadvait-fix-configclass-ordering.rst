Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.configclass.configclass` reordering class attributes when type
  annotations are provided on only some of them. Annotated attributes no longer jump ahead of
  non-annotated ones, so the resulting field order now matches the declaration order. This
  is important for configuration classes where the attribute order is meaningful, such as
  :class:`~isaaclab.scene.InteractiveSceneCfg`.
