* Added ``--checkpoint_ghosts`` to RSL-RL, rl_games, skrl, and Stable-Baselines3 playback, the Newton GL viewer's
  checkpoint visualization. ``--num_envs`` is the total simulated and is split evenly among the checkpoints shown: the
  reference starts with all of them, and adding a ghost resizes every slot (32 environments give 16 each with two
  checkpoints). Plain playback allocates nothing for the feature.
* Rejected ``--checkpoint_ghosts`` for tasks whose curriculum term sets the reset state, since ghosts disable curriculum
  terms.
