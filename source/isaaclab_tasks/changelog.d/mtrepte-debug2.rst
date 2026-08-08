Fixed
^^^^^

* Fixed velocity command visualization markers clipping through the torso of humanoid robots
  (H1, G1) by setting :attr:`~isaaclab.envs.mdp.commands.UniformVelocityCommandCfg.marker_pos_offset`
  in their respective rough environment configs.
