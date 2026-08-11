Changed
^^^^^^^

* **Breaking:** Renamed the multi-agent pendulum task from
  ``Isaac-Pendulum-Direct`` to ``Isaac-Pendulum-MARL-Direct``. Update task
  selections and imports to use :class:`~isaaclab_tasks.core.pendulum.pendulum_marl_env.PendulumMARLEnv`
  and :class:`~isaaclab_tasks.core.pendulum.pendulum_marl_env_cfg.PendulumMARLEnvCfg`.
