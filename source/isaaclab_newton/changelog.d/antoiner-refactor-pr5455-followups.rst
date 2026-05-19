Changed
^^^^^^^

* Moved Newton-native actuator USD authoring out of
  ``isaaclab_newton.actuators.authoring`` (now deleted) into
  :func:`~isaaclab.sim.schemas.define_actuator_properties`. The authoring step
  is now invoked via the schema-side ``_post_spawn`` hook on
  :class:`~isaaclab.assets.AssetBaseCfg`.
* Grouped :attr:`~isaaclab_newton.physics.NewtonManager._decimation` next to
  :attr:`~isaaclab_newton.physics.NewtonManager._num_substeps` for consistency
  with related solver-stepping configuration.
