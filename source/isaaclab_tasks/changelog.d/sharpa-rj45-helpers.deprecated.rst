* Deprecated the Sharpa RJ45 task's robot bundle helpers in favor of ``resolve_robot_usd`` and
  ``verify_robot_bundle`` in ``isaaclab_assets.robots.rizon_sharpa``. The old helper calls still work.
* Deprecated ``configure_hydroelastic_physics``. Use the solver and collision declarations in
  ``GB300SharpaInsertionEnvCfg`` when configuring the RJ45 task.
