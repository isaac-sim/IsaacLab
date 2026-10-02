* Changed the Warp in-hand reorientation environment to raise :class:`NotImplementedError` for
  configurations with ``asymmetric_obs`` enabled instead of silently omitting the critic
  observations.
