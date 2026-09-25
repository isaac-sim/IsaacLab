Fixed
^^^^^

* Fixed :func:`~isaaclab.envs.mdp.observations.image` passing the sensor's ``ProxyArray`` to the
  image normalization, which left colorized semantic segmentation unscaled in ``[0, 255]`` and
  skipped the fused normalization kernel.
