Changed
^^^^^^^

* Moved the rigid-shape contact defaults in the ``lift_franka_soft`` presets
  from ``NewtonModelCfg.shape_material_ke/kd/mu`` to
  :class:`~isaaclab_newton.physics.NewtonShapeCfg` on
  ``NewtonCfg.default_shape_cfg``.

* Made the proxy-coupled MJWarp + VBD solver the default for the
  ``Isaac-Lift-Soft-Franka`` volume task, configured through named solver
  entries and an explicit rigid-to-soft proxy mapping.
