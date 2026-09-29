Fixed
^^^^^

* Fixed :func:`~isaaclab.envs.mdp.rewards.base_height_l2` returning an infinite reward for any
  environment whose own height scan contained a missed ray, since ``ray_hits_w`` reports ``inf``
  for a ray that finds nothing within its ``max_distance`` rather than a clamped value. Now only
  finite ray hits are averaged into the terrain-adjusted target height.
