Changed
^^^^^^^

* **Breaking:** Renamed the multi-agent pendulum task from
  ``Isaac-Pendulum-Direct`` to ``Isaac-Pendulum-MARL-Direct``. Update task
  selections and imports to use :class:`~isaaclab_tasks.core.pendulum.PendulumMARLEnv`
  and :class:`~isaaclab_tasks.core.pendulum.PendulumMARLEnvCfg`.
* Changed ``Isaac-Pendulum-MARL-Direct`` to give both agents a shared team
  reward aligned with upright balancing. Retrain policies created with the
  previous split per-agent rewards.
* Changed ``Isaac-Pendulum-MARL-Direct`` to support the ``newton_mjwarp``
  physics preset. Select it with ``presets=newton_mjwarp``.
