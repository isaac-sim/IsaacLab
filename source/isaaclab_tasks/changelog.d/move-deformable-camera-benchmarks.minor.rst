Changed
^^^^^^^

* Moved the ``Isaac-Lift-Cable-Franka-Camera``, ``Isaac-Lift-Cloth-Franka-Camera``, and
  ``Isaac-Lift-Soft-Franka-Camera`` tasks into ``isaaclab_tasks.benchmark.franka_deformable_camera``.
  The task ids are unchanged, so registry-based loading needs no changes. Code that imports their environment or
  RSL-RL configuration classes directly must update to the new benchmark package paths.
