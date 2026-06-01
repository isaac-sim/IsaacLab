Changed
^^^^^^^

* **Breaking:** Restructured :mod:`isaaclab_tasks` into a flat ``core`` / ``contrib`` layout.
  Task families now live directly under :mod:`isaaclab_tasks.core` (maintained tasks) or
  :mod:`isaaclab_tasks.contrib` (contributed tasks), replacing the previous
  ``direct`` / ``manager_based`` workflow grouping and the ``classic`` / ``manipulation`` /
  ``locomotion`` domain sub-packages. Registered Gym environment IDs are unchanged, so
  ``gym.make("Isaac-...")`` calls continue to work; only Python import paths changed.
  Update imports such as
  ``from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg`` to
  ``from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg``, and
  ``from isaaclab_tasks.manager_based.manipulation.stack...`` to
  ``from isaaclab_tasks.contrib.stack...``. Tasks that exist in both workflows are now
  disambiguated by a workflow prefix (e.g. :mod:`isaaclab_tasks.core.manager_cartpole` and
  :mod:`isaaclab_tasks.core.direct_cartpole`).

Removed
^^^^^^^

* Removed the ``Isaac-Quadcopter-Direct-v0``, ``Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0``,
  ``Isaac-Repose-Cube-Allegro-NoVelObs-v0`` and ``Isaac-Repose-Cube-Allegro-NoVelObs-Play-v0``
  environments.
