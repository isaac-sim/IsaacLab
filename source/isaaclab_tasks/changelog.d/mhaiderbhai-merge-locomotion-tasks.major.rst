Changed
^^^^^^^

* **Breaking:** Grouped the ant and humanoid locomotion tasks under a new
  :mod:`isaaclab_tasks.core.locomotion` package, each as a subpackage
  (:mod:`isaaclab_tasks.core.locomotion.ant` and :mod:`isaaclab_tasks.core.locomotion.humanoid`).
  The two subpackages share the direct-workflow base environment
  (:mod:`isaaclab_tasks.core.locomotion.locomotion_direct_env`, previously
  ``isaaclab_tasks.core.direct_locomotion.locomotion_env``) and the manager-workflow MDP terms
  (:mod:`isaaclab_tasks.core.locomotion.mdp`, previously
  ``isaaclab_tasks.core.manager_humanoid.mdp``).
* **Breaking:** Merged the direct-workflow and manager-based-workflow ant and humanoid task
  packages into the per-task subpackages above; the former ``isaaclab_tasks.core.direct_ant``,
  ``isaaclab_tasks.core.manager_ant``, ``isaaclab_tasks.core.direct_humanoid`` and
  ``isaaclab_tasks.core.manager_humanoid`` packages were removed. Module files now carry a
  ``_direct_`` or ``_manager_`` infix to disambiguate the two workflows. Update imports, e.g.:

  * ``from isaaclab_tasks.core.direct_ant.ant_env import AntEnv`` →
    ``from isaaclab_tasks.core.locomotion.ant.ant_direct_env import AntEnv``.
  * ``from isaaclab_tasks.core.manager_humanoid.humanoid_env_cfg import HumanoidEnvCfg`` →
    ``from isaaclab_tasks.core.locomotion.humanoid.humanoid_manager_env_cfg import HumanoidEnvCfg``.

  The near-identical per-workflow ``rsl_rl_ppo_cfg`` modules were consolidated; each subpackage's
  ``agents.rsl_rl_ppo_cfg`` now exposes a manager-based runner cfg (:class:`AntPPORunnerCfg` /
  :class:`HumanoidPPORunnerCfg`) and a direct-workflow subclass (:class:`AntDirectPPORunnerCfg` /
  :class:`HumanoidDirectPPORunnerCfg`).
* **Breaking:** Renamed the ant and humanoid Gym environment IDs to drop the ``-v0`` version suffix
  and mark the direct-workflow tasks with an explicit ``-Direct`` suffix. The manager-based workflow
  is the default and carries no workflow suffix. Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Ant-Direct-v0`` → ``Isaac-Ant-Direct``.
  * ``Isaac-Ant-v0`` → ``Isaac-Ant``.
  * ``Isaac-Humanoid-Direct-v0`` → ``Isaac-Humanoid-Direct``.
  * ``Isaac-Humanoid-v0`` → ``Isaac-Humanoid``.
* **Breaking:** Renamed the cart double pendulum Gym environment ID to drop the ``-v0`` version
  suffix. Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Cart-Double-Pendulum-Direct-v0`` → ``Isaac-Cart-Double-Pendulum-Direct``.

Removed
^^^^^^^

* Removed the unused ``rew_scale_cart_pos`` field from
  :class:`~isaaclab_tasks.core.cart_double_pendulum.cart_double_pendulum_env_cfg.CartDoublePendulumEnvCfg`.
  It defaulted to ``0`` and was never applied to any reward term, so removing it does not change
  training behavior.
