Fixed
^^^^^

* Fixed the Pink IK action
  (:class:`~isaaclab.envs.mdp.actions.pink_task_space_actions.PinkInverseKinematicsAction`) applying zero
  joint-effort targets instead of the gravity compensation forces on fixed-base articulations when
  ``enable_gravity_compensation`` is enabled.
