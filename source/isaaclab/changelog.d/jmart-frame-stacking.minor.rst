Added
^^^^^

* Added :class:`~isaaclab.envs.mdp.observations.stacked_image`, a stateful
  :class:`~isaaclab.managers.ManagerTermBase` that channel-stacks the last ``N`` frames
  from a camera sensor. Manager-based environments can reference it in observation cfg
  to add explicit temporal information for camera-based RL tasks whose renderer doesn't
  supply implicit temporal data (e.g., Newton Warp).
