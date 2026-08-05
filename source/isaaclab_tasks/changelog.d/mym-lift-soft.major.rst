Added
^^^^^

* Added a deformable COM pose command, dense deformable rewards with success logging, a gravity
  curriculum, and ``joint`` and ``ik`` action presets to the Franka soft-beam and cloth lift tasks.

Changed
^^^^^^^

* **Breaking:** Changed the default action space to relative joint-position control. Use
  ``presets=ik`` for task-space inverse-kinematics control; integrations using the cloth
  environments' previous absolute joint targets must update their actions.
* **Breaking:** Changed the non-camera ``rsl_rl`` experiment name from ``franka_deformable`` to
  ``franka_soft``. Update log and checkpoint paths that refer to ``logs/rsl_rl/franka_deformable``.
* Re-tuned the robot, scenes, contact handling, control rate, and ``rsl_rl`` configuration for
  gravity-based training.

Removed
^^^^^^^

* **Breaking:** Removed :func:`~isaaclab_tasks.core.lift.mdp.deformable_lifted`. Use
  :func:`~isaaclab_tasks.core.lift.mdp.deformable_lifting` and set the required ``std``.
* **Breaking:** Removed :func:`~isaaclab_tasks.core.lift.mdp.deformable_com_goal_distance`. Use
  :class:`~isaaclab_tasks.core.lift.mdp.DeformableComGoalDistance` and set the required
  ``success_threshold``.
* **Breaking:** Removed :func:`~isaaclab_tasks.core.lift.mdp.deformable_outside_table_bounds`. Use
  :func:`~isaaclab_tasks.core.lift.mdp.deformable_outside_bounds` and set the required ``z_bounds``.
* **Breaking:** Removed :func:`~isaaclab_tasks.core.lift.mdp.deformable_com_below_minimum`. Use
  :func:`~isaaclab_tasks.core.lift.mdp.deformable_outside_bounds` with appropriate ``z_bounds`` for
  workspace termination.
* **Breaking:** Removed the state-machine demo's video arguments. Use ``--num_steps`` to control
  the finite demo and an external capture workflow to record it.
* **Breaking:** Removed the unsupported ``ovphysx`` preset from the Franka soft-beam and cloth lift
  environments. Use ``isaacsim_physx`` for the soft-beam task or
  ``newton_mjwarp_vbd_proxy`` for either task.
