* Fixed :attr:`~isaaclab.managers.ActionTermCfg.clip` for the task-space action terms.
  :class:`~isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction` resolved the ranges against
  the joint names and indexed the task-space action vector with joint indices (an index error, or joint
  limits applied to pose components), and :class:`~isaaclab.envs.mdp.actions.OperationalSpaceControllerAction`
  ignored the setting. Both now match the keys against task-space action names (``pos_x``, ``rot_x``,
  ``force_x``, ``stiffness_x``, ...), exposed as :attr:`action_names`, and reject keys that match nothing or
  a quaternion component.
