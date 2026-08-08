Fixed
^^^^^

* Added :attr:`~isaaclab.envs.mdp.commands.UniformVelocityCommandCfg.marker_pos_offset` to
  :class:`~isaaclab.envs.mdp.commands.UniformVelocityCommandCfg` so the velocity visualization
  markers can be repositioned per robot. The default ``(0.0, 0.0, 0.5)`` preserves existing
  behavior; humanoid configs override this to avoid arrows clipping through the robot torso.
