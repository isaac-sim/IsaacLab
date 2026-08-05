Added
^^^^^

* Added deformable-specific commands, observations, rewards, events, terminations, and curricula,
  plus ``joint`` and ``ik`` action presets, to the Franka soft-beam and cloth lift environments.

Changed
^^^^^^^

* **Breaking:** Changed the default action space to relative joint-position control. Use
  ``presets=ik`` for task-space inverse-kinematics control; integrations using the cloth
  environments' previous absolute joint targets must update their actions.
* **Breaking:** Changed the non-camera ``rsl_rl`` experiment name from ``franka_deformable`` to
  ``franka_soft``. Update log and checkpoint paths that refer to ``logs/rsl_rl/franka_deformable``.
* Re-tuned the robot, scenes, contact handling, control rate, and ``rsl_rl`` configuration for stable
  gravity-based training across the supported physics backends.

Removed
^^^^^^^

* **Breaking:** Removed the unsupported ``ovphysx`` preset from the Franka soft-beam and cloth lift
  environments. Use ``isaacsim_physx`` for the soft-beam task or ``newton_mjwarp_vbd_proxy`` for either task.
* **Breaking:** Removed :func:`~isaaclab_tasks.core.lift.mdp.deformable_lifted`. Use
  :func:`~isaaclab_tasks.core.lift.mdp.deformable_lifting` for a smooth ``tanh``-shaped lifting reward.
