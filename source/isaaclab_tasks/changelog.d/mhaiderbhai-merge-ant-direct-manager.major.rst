Changed
^^^^^^^

* **Breaking:** Merged the direct-workflow and manager-based-workflow ant task packages
  (``isaaclab_tasks.core.direct_ant`` and ``isaaclab_tasks.core.manager_ant``) into a single
  flat :mod:`isaaclab_tasks.core.ant` package. Module files now carry a ``_direct_`` or
  ``_manager_`` infix to disambiguate the two workflows. Update imports such as
  ``from isaaclab_tasks.core.direct_ant.ant_env import AntEnv`` to
  ``from isaaclab_tasks.core.ant.ant_direct_env import AntEnv``, and
  ``from isaaclab_tasks.core.manager_ant.ant_env_cfg import AntEnvCfg`` to
  ``from isaaclab_tasks.core.ant.ant_manager_env_cfg import AntEnvCfg``. The near-identical
  per-workflow ``rsl_rl_ppo_cfg`` modules were consolidated into a single
  :mod:`isaaclab_tasks.core.ant.agents.rsl_rl_ppo_cfg` module exposing
  :class:`~isaaclab_tasks.core.ant.agents.rsl_rl_ppo_cfg.AntPPORunnerCfg` (manager-based) and
  :class:`~isaaclab_tasks.core.ant.agents.rsl_rl_ppo_cfg.AntDirectPPORunnerCfg` (direct).
* **Breaking:** Renamed the ant Gym environment IDs to drop the ``-v0`` version suffix and
  mark the direct-workflow task with an explicit ``-Direct`` suffix. The manager-based workflow
  is the default and carries no workflow suffix. Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Ant-Direct-v0`` → ``Isaac-Ant-Direct``.
  * ``Isaac-Ant-v0`` → ``Isaac-Ant``.
* **Breaking:** Renamed the cart double pendulum Gym environment ID to drop the ``-v0`` version
  suffix. Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Cart-Double-Pendulum-Direct-v0`` → ``Isaac-Cart-Double-Pendulum-Direct``.
