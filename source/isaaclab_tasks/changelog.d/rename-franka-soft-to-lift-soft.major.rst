Changed
^^^^^^^

* **Breaking:** Changed the non-camera soft-beam ``rsl_rl`` experiment name from ``franka_soft`` to
  ``lift_soft``, matching the ``lift_cloth`` and ``lift_cable`` naming of the sibling tasks. This
  also moves the logs of the contributed ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling`` task,
  which reuses the same runner config. Existing checkpoints remain loadable: update log and
  checkpoint paths that refer to ``logs/rsl_rl/franka_soft``, or pass
  ``--experiment_name franka_soft``.
