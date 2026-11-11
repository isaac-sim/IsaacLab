* Added ``Isaac-Cartpole-MDP2``, the Cartpole task written with :mod:`isaaclab_experimental.mdp2` and run by
  :class:`~isaaclab_experimental.envs.MDP2RLEnv` on Newton, in :mod:`isaaclab_tasks.mdp2`.
* Added ``Isaac-Velocity-Flat-UnitreeGo2-MDP2``, the Unitree Go2 flat velocity task written with
  :mod:`isaaclab_experimental.mdp2`, with its startup randomization and heading commands.
* Added ``Isaac-Velocity-Flat-WalkStand-UnitreeGo2-MDP2``, which trains one Go2 policy in a scene where half of the
  worlds follow velocity commands and the other half stand still, with their own rewards.
