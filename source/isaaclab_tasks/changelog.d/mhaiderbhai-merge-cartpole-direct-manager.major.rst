Changed
^^^^^^^

* **Breaking:** Merged the direct-workflow and manager-based-workflow cartpole task packages
  (``isaaclab_tasks.core.direct_cartpole`` and ``isaaclab_tasks.core.manager_cartpole``) into a
  single flat :mod:`isaaclab_tasks.core.cartpole` package. Module files now carry a ``_direct_``
  or ``_manager_`` infix to disambiguate the two workflows. Update imports such as
  ``from isaaclab_tasks.core.direct_cartpole.cartpole_env import CartpoleEnv`` to
  ``from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv``, and
  ``from isaaclab_tasks.core.manager_cartpole.cartpole_env_cfg import CartpoleEnvCfg`` to
  ``from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg``.
* **Breaking:** Renamed the cartpole Gym environment IDs to drop the ``-v0`` version suffix
  and mark the direct-workflow tasks with an explicit ``-Direct`` suffix. The manager-based
  workflow is the default and carries no workflow suffix. Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Cartpole-Direct-v0`` → ``Isaac-Cartpole-Direct``.
  * ``Isaac-Cartpole-Camera-Direct-v0`` → ``Isaac-Cartpole-Camera-Direct``.
  * ``Isaac-Cartpole-v0`` → ``Isaac-Cartpole``.
  * ``Isaac-Cartpole-Camera-v0`` → ``Isaac-Cartpole-Camera``.
  * ``Isaac-Cartpole-Showcase-Direct-v0`` → ``Isaac-Cartpole-Showcase-Direct``.
  * ``Isaac-Cartpole-Camera-Showcase-Direct-v0`` → ``Isaac-Cartpole-Camera-Showcase-Direct``.
