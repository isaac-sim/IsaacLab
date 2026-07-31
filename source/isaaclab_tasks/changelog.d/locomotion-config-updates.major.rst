Changed
^^^^^^^

* **Breaking:** Moved the Spot velocity environment from ``isaaclab_tasks.core``
  to ``isaaclab_tasks.contrib`` and renamed its task ID from
  ``Isaac-Velocity-Flat-Spot`` to ``IsaacContrib-Velocity-Flat-Spot``. Update
  Python imports and ``gym.make`` / ``--task`` arguments to use the contributed
  path and ID.
* Unified the locomotion velocity physics presets to expose automatic PhysX
  selection and the supported concrete PhysX and Newton backends.
* Tuned the Newton and PhysX solver settings for the locomotion velocity
  environments.
