Changed
^^^^^^^

* **Breaking:** Renamed the multi-agent pendulum task from
  ``Isaac-Pendulum-Direct`` to ``Isaac-Pendulum-MARL-Direct``. Update task
  selections and imports to use :class:`~isaaclab_tasks.core.pendulum.pendulum_marl_env.PendulumMARLEnv`
  and :class:`~isaaclab_tasks.core.pendulum.pendulum_marl_env_cfg.PendulumMARLEnvCfg`.
* Changed ``Isaac-Pendulum-MARL-Direct`` to give both agents a shared team
  reward aligned with upright balancing. Retrain policies created with the
  previous split per-agent rewards.
