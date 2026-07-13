Changed
^^^^^^^

* Moved the rigid-shape contact defaults in the ``lift_franka_soft`` presets
  from ``NewtonModelCfg.shape_material_ke/kd/mu`` to
  :class:`~isaaclab_newton.physics.NewtonShapeCfg` on
  ``NewtonCfg.default_shape_cfg``.

* Migrated ``lift_franka_soft`` (rigid + cloth variants) from
  ``DeformableNewtonCfg`` to plain :class:`~isaaclab_newton.physics.NewtonCfg`
  presets whose solver configs carry
  :class:`~isaaclab_contrib.deformable.newton_manager_cfg.NewtonModelCfg`,
  with the proxy-coupled MJWarp + VBD solver as the default, configured through
  named solver entries and an explicit rigid-to-soft proxy mapping.
