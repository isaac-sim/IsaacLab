Changed
^^^^^^^

* **Breaking:** Split the camera ``rsl_rl`` experiment name ``franka_deformable_camera``, which
  ``Isaac-Lift-Soft-Franka-Camera`` and ``Isaac-Lift-Cloth-Franka-Camera`` previously shared, into
  ``lift_soft_camera`` and ``lift_cloth_camera``. Because both tasks have identical observation and
  action shapes, a run of one task could silently load the other's checkpoint when replaying without
  an explicit ``--checkpoint``. Move runs out of ``logs/rsl_rl/franka_deformable_camera`` into the
  new per-task directories, using the ``task`` field of each run's ``run_manifest.json`` to tell them
  apart.

Added
^^^^^

* Added :class:`~isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg.FrankaClothCameraPPORunnerCfg`
  so ``Isaac-Lift-Cloth-Franka-Camera`` logs to its own experiment directory.
