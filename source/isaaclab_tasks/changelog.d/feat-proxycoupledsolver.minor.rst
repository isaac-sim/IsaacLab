Changed
^^^^^^^

* **Breaking:** Renamed the deformable scene entity and its MDP terms in
  ``lift_franka_soft`` from ``deformable`` to ``object`` to align with the
  rigid lift task. Affects ``Isaac-Lift-Soft-Franka-v0`` and the cloth
  variant: scene entry ``scene.deformable`` -> ``scene.object``, command
  ``deformable_pose`` -> ``object_pose``, and MDP functions
  ``deformable_ee_distance``, ``deformable_lifted``,
  ``deformable_com_goal_distance``, ``deformable_com_in_robot_root_frame``,
  ``deformable_com_below_minimum``, ``deformable_outside_table_bounds``,
  ``DeformableSampledPointsInRobotRootFrame`` -> ``object_*`` /
  ``ObjectSampledPointsInRobotRootFrame``. Update env configs, checkpoints,
  and RL configs accordingly.

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
