Changed
^^^^^^^

* **Breaking:** Consolidated the Franka cabinet (open-drawer) tasks into the single
  :mod:`isaaclab_tasks.core.cabinet` package. The former ``isaaclab_tasks.core.franka_cabinet``
  direct task moved in: the env ``FrankaCabinetEnv`` became the robot-agnostic
  :class:`~isaaclab_tasks.core.cabinet.cabinet_direct_env.CabinetDirectEnv` (at the package root,
  alongside the manager-based :class:`~isaaclab_tasks.core.cabinet.cabinet_env_cfg.CabinetEnvCfg`).
  Its config ``FrankaCabinetEnvCfg`` (a ``DirectRLEnvCfg``) was split into a robot-agnostic base
  :class:`~isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg.CabinetDirectEnvCfg` (at the package
  root, with the robot left unset) and the Franka subclass
  :class:`~isaaclab_tasks.core.cabinet.config.franka.cabinet_direct_env_cfg.FrankaCabinetDirectEnvCfg`
  (which supplies the arm). The split also avoids clashing with the manager-based
  ``FrankaCabinetEnvCfg`` in :mod:`~isaaclab_tasks.core.cabinet.config.franka.joint_pos_env_cfg`.
* **Breaking:** Renamed the Gym environment IDs: dropped the ``-v0`` version suffix and unified the
  direct task under the manager's ``Open-Drawer-Franka`` name with a ``-Direct`` suffix. Update
  ``gym.make`` / ``--task`` calls:

  * ``Isaac-Open-Drawer-Franka-v0`` → ``Isaac-Open-Drawer-Franka``.
  * ``Isaac-Open-Drawer-Franka-Play-v0`` → ``Isaac-Open-Drawer-Franka-Play``.
  * ``Isaac-Franka-Cabinet-Direct-v0`` → ``Isaac-Open-Drawer-Franka-Direct``.
* **Breaking:** Merged the direct and manager agent configs into
  ``isaaclab_tasks.core.cabinet.config.franka.agents``. The colliding ``rl_games`` / ``skrl`` configs
  carry a ``_direct_`` / ``_manager_`` infix (e.g. ``rl_games_manager_ppo_cfg.yaml`` /
  ``rl_games_direct_ppo_cfg.yaml``), and the ``rsl_rl`` runner configs ``CabinetPPORunnerCfg``
  (manager) and ``FrankaCabinetPPORunnerCfg`` (direct) now live together in
  ``cabinet.config.franka.agents.rsl_rl_ppo_cfg``.

Fixed
^^^^^

* Removed an unused ``joint_positions`` parameter from the cabinet direct environment's reward
  computation.
