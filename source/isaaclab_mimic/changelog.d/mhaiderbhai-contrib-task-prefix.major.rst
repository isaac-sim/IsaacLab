Changed
^^^^^^^

* **Breaking:** Renamed the registered Mimic/Skillgen environment IDs to use the ``IsaacContrib-``
  prefix instead of ``Isaac-`` and dropped the trailing ``-v0`` version suffix, matching the
  contributed task naming convention. Update ``gym.make`` / ``--task`` calls accordingly, for example
  ``Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0`` → ``IsaacContrib-Stack-Cube-Bin-Franka-IK-Rel-Mimic``.
