Fixed
^^^^^

* Fixed :func:`~isaaclab_rl.rsl_rl.utils.handle_deprecated_rsl_rl_cfg` leaving ``obs_groups``
  unset on ``OnPolicyRunner`` configs for rsl-rl-lib >= 4.0.0, which made every play and
  evaluation script hang during runner initialization.
