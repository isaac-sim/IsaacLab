Added
^^^^^

* Added the ``Isaac-Lift-Cube-SO101`` and ``Isaac-Lift-Cube-SO101-Play`` manager-based
  environments for lifting a cube to a commanded pose with the SO-101 arm, mirroring the
  Franka cube-lift task configuration.
* Added :func:`~isaaclab_tasks.core.lift.mdp.object_grasped` rewarding a pinch grasp, and
  an optional ``grasp_params`` parameter on :func:`~isaaclab_tasks.core.lift.mdp.object_is_lifted`
  and :class:`~isaaclab_tasks.core.lift.mdp.object_goal_distance` that gates the reward on
  the object being held in the gripper jaws (off by default).
