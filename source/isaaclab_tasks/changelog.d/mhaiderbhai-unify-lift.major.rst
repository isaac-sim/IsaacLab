Changed
^^^^^^^

* **Breaking:** Merged the ``isaaclab_tasks.core.dexsuite`` package into
  :mod:`isaaclab_tasks.core.lift`, so the dexterous lift and reorient tasks share one package and
  MDP module. Task IDs and training behavior were unchanged, but environment configuration entry
  points moved and the ``Dexsuite`` class-name prefix was removed. For example,
  ``isaaclab_tasks.core.dexsuite.config.franka.dexsuite_franka_env_cfg:DexsuiteFrankaLiftEnvCfg``
  became ``isaaclab_tasks.core.lift.config.franka.franka_env_cfg:FrankaLiftEnvCfg``.
* **Breaking:** Moved the tutorial single-cube Franka lift task to :mod:`isaaclab_tasks.contrib.lift`.
  Use ``--task IsaacContrib-Lift-Cube-Franka`` instead of
  ``--task Isaac-Lift-Cube-Franka``. The rigid tutorial MDP terms moved with it; deformable
  Lift terms remain in :mod:`isaaclab_tasks.core.lift.mdp`.
* **Breaking:** Renamed the dexterous RSL-RL experiment directories from
  ``dexsuite_franka`` and ``dexsuite_kuka_allegro*`` to ``lift_franka`` and
  ``lift_kuka_allegro*``. Update existing checkpoint paths under ``logs/rsl_rl/dexsuite_*``.
