Changed
^^^^^^^

* **Breaking:** Renamed the multi-agent pendulum task from
  ``Isaac-Pendulum-Direct`` to ``Isaac-Pendulum-MARL-Direct``. Update task
  selections and imports to use :class:`~isaaclab_tasks.core.pendulum.PendulumMARLEnv`
  and :class:`~isaaclab_tasks.core.pendulum.PendulumMARLEnvCfg`.
* Changed ``Isaac-Pendulum-MARL-Direct`` to give both agents a shared team
  reward aligned with upright balancing, including a bonus when both links
  enter the success cone. Retrain policies created with the previous split
  per-agent rewards.
* Changed ``Isaac-Pendulum-MARL-Direct`` to default to ``newton_mjwarp`` and
  support the ``newton_kamino`` and ``ovphysx`` physics backends. Select the
  previous Isaac Sim PhysX default with ``physics=isaacsim_physx``.
* Changed ``Isaac-Pendulum-MARL-Direct`` to use ``0.05 kg m^2`` of armature
  on the lower pendulum joint, aligning its full-scale ``50 N m`` response
  between PhysX and MJWarp. Retrain policies created without the armature.
