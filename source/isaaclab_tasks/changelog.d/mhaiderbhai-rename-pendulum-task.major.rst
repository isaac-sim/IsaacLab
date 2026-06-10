Changed
^^^^^^^

* **Breaking:** Renamed the cart double pendulum task to ``pendulum``. The package moved from
  ``isaaclab_tasks.core.cart_double_pendulum`` to :mod:`isaaclab_tasks.core.pendulum`, the env
  ``CartDoublePendulumEnv`` / ``CartDoublePendulumEnvCfg`` became
  :class:`~isaaclab_tasks.core.pendulum.pendulum_env.PendulumEnv` /
  :class:`~isaaclab_tasks.core.pendulum.pendulum_env_cfg.PendulumEnvCfg`, and the Gym environment ID
  ``Isaac-Cart-Double-Pendulum-Direct`` became ``Isaac-Pendulum-Direct``. The
  ``CART_DOUBLE_PENDULUM_CFG`` robot asset in :mod:`isaaclab_assets` is unchanged.
