Changed
^^^^^^^

* **Breaking:** Moved the Digit velocity environments from ``isaaclab_tasks.core``
  to ``isaaclab_tasks.contrib`` and renamed their task IDs from
  ``Isaac-Velocity-{Flat,Rough}-Digit`` to
  ``IsaacContrib-Velocity-{Flat,Rough}-Digit``. Update Python imports and
  ``gym.make`` / ``--task`` arguments to use the contributed paths and IDs.
  Digit velocity and loco-manip environments now support only the ``physx``
  physics preset.
