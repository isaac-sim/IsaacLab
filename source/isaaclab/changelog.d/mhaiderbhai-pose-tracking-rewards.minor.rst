Added
^^^^^

* Added the pose-tracking reward terms :func:`~isaaclab.envs.mdp.rewards.position_command_error`,
  :func:`~isaaclab.envs.mdp.rewards.position_command_error_tanh` and
  :func:`~isaaclab.envs.mdp.rewards.orientation_command_error` to the shared MDP reward terms. They
  track a body pose against a pose command and complement the existing velocity-tracking terms. The
  reach task previously defined these locally.
