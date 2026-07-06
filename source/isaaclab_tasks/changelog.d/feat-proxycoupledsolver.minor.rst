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

* Migrated ``lift_franka_soft`` (rigid + cloth variants) from
  ``DeformableNewtonCfg`` to
  :class:`~isaaclab_contrib.deformable.newton_manager_cfg.CoupledNewtonCfg`
  with the proxy-coupled MJWarp + VBD solver as the default, configured through
  named solver entries and an explicit rigid-to-soft proxy mapping.
