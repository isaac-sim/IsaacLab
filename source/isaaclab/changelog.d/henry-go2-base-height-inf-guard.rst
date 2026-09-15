Fixed
^^^^^

* Fixed :func:`~isaaclab.envs.mdp.rewards.base_height_l2` returning an infinite reward for every
  environment in the batch whenever a single height-scanner ray missed the terrain within its
  ``max_distance``, since ``ray_hits_w`` reports ``inf`` for a miss rather than a clamped value.
  Now only finite ray hits are averaged into the terrain-adjusted target height.
