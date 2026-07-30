Changed
^^^^^^^

* **Breaking:** Moved the Spot velocity environment from ``isaaclab_tasks.core``
  to ``isaaclab_tasks.contrib`` and renamed its task ID from
  ``Isaac-Velocity-Flat-Spot`` to ``IsaacContrib-Velocity-Flat-Spot``. Update
  Python imports and ``gym.make`` / ``--task`` arguments to use the contributed
  path and ID.
* Unified the physics preset layout across the locomotion velocity
  configurations so each ``PhysicsCfg`` declares ``physx`` as the primary
  backend and aliases ``default = physx``.
* Tuned the Newton and PhysX solver settings for the locomotion velocity
  environments.
